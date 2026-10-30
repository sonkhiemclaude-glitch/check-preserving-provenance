"""Bedrock bearer authentication with explicit routes; no model/provider fallback.

Contracts: AWS Bedrock Chat Completions and Converse APIs, checked 2026-10-24.
Uses the standard-library HTTP client; it never sends credentials in JSON or logs.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shlex
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ProviderError(RuntimeError):
    """Contains only a safe error code, never provider bodies or credentials."""

    def __init__(self, code: str, status: int | None = None):
        self.code, self.status = code, status
        super().__init__(f"{code}" + (f" (HTTP {status})" if status else ""))


def read_env(path: str | Path | None) -> dict[str, str]:
    """Read ordinary .env assignments without sourcing commands or interpolating values.

    Existing process variables take precedence. Supports the user's Bedrock_API
    spelling as well as the documented AWS_BEARER_TOKEN_BEDROCK name.
    """
    values: dict[str, str] = {}
    if path and Path(path).is_file():
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ")
            key, sep, raw = line.partition("=")
            key = key.strip()
            if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"invalid .env assignment on line {number}")
            if key in values:
                raise ValueError(f"duplicate .env field {key}")
            try:
                parts = shlex.split(raw, comments=True, posix=True)
            except ValueError as exc:
                raise ValueError(f"invalid .env quoting on line {number}") from exc
            values[key] = " ".join(parts)
    values.update(os.environ)
    return values


@dataclass(frozen=True)
class ProviderConfig:
    model_id: str
    base_url: str
    api_mode: str = "bedrock-chat"
    region: str = ""
    api_key: str = field(default="", repr=False, compare=False)
    temperature: float | None = 0.0
    reasoning_effort: str | None = None
    token_parameter: str = "max_tokens"
    max_output_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 2
    max_requests: int = 80
    max_total_tokens: int = 100_000
    max_message_chars: int = 120_000

    @classmethod
    def from_env(cls, env: dict[str, str]) -> ProviderConfig:
        mode = env.get("BEDROCK_API_MODE", "bedrock-chat").strip()
        model = env.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b").strip()
        reasoning_model = model.removeprefix("openai.").startswith(("gpt-5", "gpt-6"))
        oss_model = model.removeprefix("openai.").startswith("gpt-oss-")
        temperature = env.get("E5_TEMPERATURE", "default" if reasoning_model else "0").strip()
        region = env.get("AWS_REGION", env.get("AWS_DEFAULT_REGION", env.get("BEDROCK_REGION", "us-east-1"))).strip()
        base = env.get("BEDROCK_BASE_URL", "").strip().rstrip("/")
        if not base and region and mode == "bedrock-converse":
            base = f"https://bedrock-runtime.{region}.amazonaws.com"
        elif not base and region and mode == "bedrock-chat":
            base = f"https://bedrock-mantle.{region}.api.aws" + ("/openai/v1" if reasoning_model else "/v1")
        key = (env.get("AWS_BEARER_TOKEN_BEDROCK") or env.get("BEDROCK_API_KEY") or env.get("Bedrock_API") or "")
        effort = "none" if reasoning_model else "low" if oss_model else ""
        token_parameter = "max_completion_tokens" if reasoning_model or oss_model else "max_tokens"
        return cls(model_id=model, base_url=base, api_mode=mode, region=region,
                   api_key=key, temperature=None if temperature in ("", "default") else float(temperature),
                   reasoning_effort=env.get("E5_REASONING_EFFORT", effort) or None,
                   token_parameter=env.get("E5_TOKEN_PARAMETER", token_parameter),
                   max_output_tokens=int(env.get("E5_MAX_OUTPUT_TOKENS", "1024")),
                   timeout=float(env.get("E5_TIMEOUT_SECONDS", "60")), max_retries=int(env.get("E5_MAX_RETRIES", "2")),
                   max_requests=int(env.get("E5_MAX_REQUESTS", "80")),
                   max_total_tokens=int(env.get("E5_MAX_TOTAL_TOKENS", "100000")))

    def validate(self, require_key: bool = True) -> None:
        if self.api_mode not in ("bedrock-chat", "bedrock-converse", "openai-compatible"):
            raise ValueError("BEDROCK_API_MODE must be bedrock-chat, bedrock-converse or openai-compatible")
        if not self.model_id:
            raise ValueError("missing BEDROCK_MODEL_ID; use the exact ID supplied by the endpoint owner")
        if not self.base_url:
            raise ValueError("missing BEDROCK_BASE_URL (or AWS_REGION for the native Bedrock runtime endpoint)")
        url = urlsplit(self.base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("BEDROCK_BASE_URL must be an HTTPS base URL without credentials, query or fragment")
        if self.api_mode.startswith("bedrock"):
            if not re.fullmatch(r"bedrock-(runtime\.[a-z0-9-]+\.amazonaws\.com|mantle\.[a-z0-9-]+\.api\.aws)",
                                url.hostname):
                raise ValueError("native Bedrock mode requires an AWS endpoint; choose openai-compatible for a gateway")
        if self.api_mode == "bedrock-converse" and (url.path not in ("", "/") or "bedrock-runtime." not in url.hostname):
            raise ValueError("Converse requires the bedrock-runtime host without an /openai/v1 prefix")
        if self.base_url.endswith(("/chat/completions", "/responses")):
            raise ValueError("BEDROCK_BASE_URL must be the base URL, without /chat/completions or /responses")
        if require_key and not self.api_key:
            raise ValueError("missing AWS_BEARER_TOKEN_BEDROCK, BEDROCK_API_KEY or Bedrock_API")
        if any(c in self.api_key for c in ("\r", "\n")):
            raise ValueError("API key must be on one line")
        if self.temperature is not None and (not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ValueError("temperature must be finite and in [0, 2], or default")
        if self.token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("token parameter must be max_tokens or max_completion_tokens")
        if self.reasoning_effort not in (None, "none", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("unsupported reasoning effort")
        if self.api_mode == "bedrock-converse" and self.reasoning_effort is not None:
            raise ValueError("reasoning effort mapping is model-specific; use the Chat Completions route")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if not 0 <= self.max_retries <= 5 or not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("max retries must be 0..5 and output tokens 1..8192")
        if self.max_requests < 1 or self.max_total_tokens < 1:
            raise ValueError("request and token limits must be positive")

    def public(self) -> dict:
        return {"api_mode": self.api_mode, "model_id": self.model_id, "base_url": self.base_url,
                "region": self.region, "temperature": self.temperature, "reasoning_effort": self.reasoning_effort,
                "token_parameter": self.token_parameter, "max_output_tokens": self.max_output_tokens,
                "timeout": self.timeout, "max_retries": self.max_retries, "max_requests": self.max_requests,
                "max_total_tokens": self.max_total_tokens, "model_seed_supported": False}


@dataclass
class Completion:
    message: dict
    response: dict
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    replay: bool = False
    mock: bool = False
    request: dict | None = None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("redirect_refused", code)


def _post(url: str, payload: dict, key: str, timeout: float) -> dict:
    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
                  "Authorization": f"Bearer {key}"}, method="POST")
    # A redirect must not forward the bearer credential to another endpoint.
    with build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
        body = response.read(8 * 1024 * 1024 + 1)
    if len(body) > 8 * 1024 * 1024:
        raise ProviderError("response_too_large")
    return json.loads(body)


def _converse_payload(messages: list[dict], tools: list[dict], config: ProviderConfig) -> dict:
    system: list[dict] = []
    conversation: list[dict] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system.append({"text": message["content"]})
            continue
        content: list[dict[str, Any]] = []
        if role == "tool":
            content.append({"toolResult": {"toolUseId": message["tool_call_id"],
                                           "content": [{"text": message["content"]}]}})
            role = "user"
        else:
            if message.get("content"):
                content.append({"text": message["content"]})
            for call in message.get("tool_calls", []):
                content.append({"toolUse": {"toolUseId": call["id"], "name": call["function"]["name"],
                                            "input": json.loads(call["function"]["arguments"])}})
        if not content:
            raise ProviderError("empty_message")
        if conversation and conversation[-1]["role"] == role:
            conversation[-1]["content"].extend(content)
        else:
            conversation.append({"role": role, "content": content})
    return {"system": system, "messages": conversation,
            "inferenceConfig": {**({"temperature": config.temperature} if config.temperature is not None else {}),
                                "maxTokens": config.max_output_tokens},
            "toolConfig": {"tools": [{"toolSpec": {"name": t["function"]["name"],
                            "description": t["function"]["description"],
                            "inputSchema": {"json": t["function"]["parameters"]}}} for t in tools]}}


def _normalise(response: dict, mode: str) -> Completion:
    if mode == "bedrock-converse":
        if response.get("stopReason") in ("max_tokens", "guardrail_intervened", "content_filtered"):
            raise ProviderError("truncated_or_filtered_response")
        blocks = response["output"]["message"]["content"]
        message: dict = {"role": "assistant", "content": "\n".join(b["text"] for b in blocks if "text" in b)}
        calls = [{"id": b["toolUse"]["toolUseId"], "type": "function",
                  "function": {"name": b["toolUse"]["name"], "arguments": json.dumps(b["toolUse"]["input"])}}
                 for b in blocks if "toolUse" in b]
        usage = response.get("usage", {})
        input_tokens, output_tokens = usage.get("inputTokens", 0), usage.get("outputTokens", 0)
    else:
        choice = response["choices"][0]
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise ProviderError("truncated_or_filtered_response")
        src = choice["message"]
        if src.get("refusal"):
            raise ProviderError("model_refusal")
        message = {"role": "assistant", "content": src.get("content") or ""}
        calls = src.get("tool_calls") or []
        usage = response.get("usage", {})
        input_tokens, output_tokens = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    ids = set()
    for call in calls:
        if call.get("type") != "function" or not isinstance(call.get("id"), str) or call["id"] in ids:
            raise ProviderError("invalid_tool_call")
        ids.add(call["id"])
        if not isinstance(call["function"]["name"], str) or not isinstance(call["function"]["arguments"], str):
            raise ProviderError("invalid_tool_call")
    if calls:
        message["tool_calls"] = calls
    if not message["content"] and not calls:
        raise ProviderError("empty_response")
    return Completion(message, response, int(input_tokens), int(output_tokens))


class BedrockClient:
    def __init__(self, config: ProviderConfig, transport=None, sleep=time.sleep, audit=None):
        config.validate()
        self.config, self.transport, self.sleep = config, transport or _post, sleep
        self.requests = self.total_tokens = 0
        self.events: list[dict] = []
        self.audit = audit

    def _event(self, event: dict) -> None:
        self.events.append(event)
        if self.audit:
            self.audit(event)

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        cfg = self.config
        if len(json.dumps(messages)) > cfg.max_message_chars:
            raise ProviderError("message_budget_exceeded")
        if cfg.api_mode == "bedrock-converse":
            url = cfg.base_url + "/model/" + quote(cfg.model_id, safe="") + "/converse"
            payload = _converse_payload(messages, tools, cfg)
        else:
            url = cfg.base_url + "/chat/completions"
            payload = {"model": cfg.model_id, "messages": messages, "tools": tools,
                       cfg.token_parameter: cfg.max_output_tokens, "parallel_tool_calls": False}
            if cfg.temperature is not None:
                payload["temperature"] = cfg.temperature
            if cfg.reasoning_effort is not None:
                payload["reasoning_effort"] = cfg.reasoning_effort
        for attempt in range(cfg.max_retries + 1):
            if self.requests >= cfg.max_requests or self.total_tokens >= cfg.max_total_tokens:
                raise ProviderError("request_or_token_budget_exhausted")
            self.requests += 1
            self._event({"attempt": self.requests, "status": "started"})
            try:
                raw = self.transport(url, payload, cfg.api_key, cfg.timeout)
                usage = raw.get("usage", {})
                input_tokens = int(usage.get("inputTokens" if cfg.api_mode == "bedrock-converse" else "prompt_tokens", 0))
                output_tokens = int(usage.get("outputTokens" if cfg.api_mode == "bedrock-converse" else "completion_tokens", 0))
                self.total_tokens += input_tokens + output_tokens
                self._event({"attempt": self.requests, "status": "ok", "input_tokens": input_tokens,
                             "output_tokens": output_tokens})
                try:
                    result = _normalise(raw, cfg.api_mode)
                except (ProviderError, KeyError, IndexError, TypeError, ValueError) as exc:
                    code = exc.code if isinstance(exc, ProviderError) else "invalid_provider_response"
                    # Keep paid usage and the actual malformed response for private diagnosis.
                    # It must never be accepted as a valid tool call or silently repaired.
                    self._event({"attempt": self.requests, "status": "response_rejected", "code": code, "response": raw})
                    raise ProviderError(code) from None
                result.request = payload
                result.attempts = attempt + 1
                return result
            except HTTPError as exc:
                self._event({"attempt": self.requests, "status": exc.code})
                retry = exc.code in (429, 500, 502, 503, 504)
                # Never include exception text, URL headers or provider error bodies.
                if not retry or attempt == cfg.max_retries:
                    raise ProviderError("http_error", exc.code) from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = min(float(retry_after), 30.0) if retry_after and retry_after.isdigit() else 2.0 ** attempt
            except (URLError, TimeoutError, socket.timeout):
                self._event({"attempt": self.requests, "status": "network_error"})
                if attempt == cfg.max_retries:
                    raise ProviderError("network_error") from None
                delay = 2.0 ** attempt
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                raise ProviderError("invalid_provider_response") from None
            self.sleep(delay + random.uniform(0, 0.2))
        raise ProviderError("retry_exhausted")

    def probe(self) -> dict:
        tool = {"type": "function", "function": {"name": "provenance_probe", "description": "Verify tool calling.",
                "parameters": {"type": "object", "properties": {"value": {"type": "integer", "enum": [7]}},
                               "required": ["value"], "additionalProperties": False}}}
        result = self.complete([{"role": "user", "content": "Call provenance_probe exactly once with value=7. "
                                 "Do not answer with ordinary text."}], [tool])
        calls = result.message.get("tool_calls", [])
        if len(calls) != 1 or calls[0]["function"]["name"] != "provenance_probe":
            raise ProviderError("tool_calling_probe_failed")
        try:
            arguments = json.loads(calls[0]["function"]["arguments"])
        except (TypeError, ValueError):
            raise ProviderError("tool_calling_probe_failed") from None
        if arguments != {"value": 7}:
            raise ProviderError("tool_calling_probe_failed")
        return {"status": "ok", "requested_model": self.config.model_id,
                "reported_model": result.response.get("model"), "tool_calling": True,
                "requests": self.requests, "tokens": self.total_tokens}

    def discover_models(self) -> dict:
        """Read the Mantle catalog. Route discovery never changes the model.

        Bedrock runtime does not implement GET /models. Mantle exposes its
        catalog on /v1/models even when inference uses /openai/v1.
        """
        cfg = self.config
        if cfg.api_mode == "bedrock-converse" or "bedrock-runtime." in (urlsplit(cfg.base_url).hostname or ""):
            raise ProviderError("models_catalog_not_supported_on_runtime")
        bases = [cfg.base_url]
        if cfg.api_mode == "bedrock-chat" and cfg.base_url.endswith("/openai/v1"):
            bases.append(cfg.base_url.removesuffix("/openai/v1") + "/v1")
        for base in bases:
            req = Request(base + "/models", headers={"Authorization": f"Bearer {cfg.api_key}"})
            try:
                with build_opener(_NoRedirect()).open(req, timeout=cfg.timeout) as response:
                    body = json.loads(response.read(2 * 1024 * 1024))
                models = sorted({row["id"] for row in body["data"] if isinstance(row.get("id"), str)})
                return {"catalog_base_url": base, "requested_model": cfg.model_id,
                        "requested_model_listed": cfg.model_id in models, "model_ids": models}
            except HTTPError as exc:
                if exc.code == 404 and base != bases[-1]:
                    continue
                raise ProviderError("catalog_http_error", exc.code) from None
            except (URLError, TimeoutError, socket.timeout):
                raise ProviderError("catalog_network_error") from None
            except (KeyError, TypeError, ValueError):
                raise ProviderError("invalid_models_catalog") from None
        raise ProviderError("models_catalog_not_found")
