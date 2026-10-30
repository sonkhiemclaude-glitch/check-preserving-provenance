"""Local numeric-analysis tools and real execution provenance for E5.

The LLM chooses tool calls. Tools operate on allowlisted CSV benchmarks;
no generated Python or shell is executed. Site assignment is simulated and
explicit in the manifest; model responses and tool execution are captured.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.schemas import Graph
from ..data.spec import Constraint, Spec, ThresholdRule, TimedRule, VocabRule
from ..data.synthetic import ACT_COLS, ENT_COLS
from .provider import Completion, ProviderError


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


@dataclass
class Dataset:
    name: str
    path: Path
    target: str
    source: str
    sites: int = 8
    seed: int = 42
    partition_strategy: str = "uniform"

    def __post_init__(self):
        self.path = self.path.resolve()
        raw = self.path.read_bytes()
        self.sha256 = digest(raw)
        self.frame = pd.read_csv(self.path)
        if not self.source or self.sites < 2 or self.sites > 32:
            raise ValueError("dataset source is required and simulated site count must be 2..32")
        if self.target not in self.frame or not pd.api.types.is_numeric_dtype(self.frame[self.target]):
            raise ValueError(f"dataset {self.name}: target must be a numeric CSV column")
        numeric = list(self.frame.select_dtypes(include="number").columns)
        self.features = [c for c in numeric if c != self.target]
        if not self.features or len(self.frame) < self.sites * 12 or self.frame[self.target].isna().any():
            raise ValueError(f"dataset {self.name}: need features, a complete target and >=12 rows per site")
        if not np.isfinite(self.frame[numeric].to_numpy(dtype=float)[~self.frame[numeric].isna().to_numpy()]).all():
            raise ValueError(f"dataset {self.name}: numeric values must be finite")
        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(len(self.frame))
        if self.partition_strategy == "uniform":
            self.indices = np.array_split(permutation, self.sites)
        elif self.partition_strategy == "weighted":
            weights = rng.lognormal(0.0, 0.7, self.sites)
            fractional = (len(self.frame) - 12 * self.sites) * weights / weights.sum()
            sizes = 12 + np.floor(fractional).astype(int)
            for i in np.argsort(-(fractional - np.floor(fractional)))[:len(self.frame) - int(sizes.sum())]:
                sizes[i] += 1
            self.indices = list(np.split(permutation, np.cumsum(sizes)[:-1]))
        else:
            raise ValueError("partition_strategy must be uniform or weighted")
        self.site_ids = [f"{self.name}-site-{i:02d}" for i in range(self.sites)]
        # CSV writer includes CRLF (one extra byte vs pandas' LF), making this
        # conservative for the serialised per-record contribution to n_bytes.
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        self.max_row_bytes = 0
        for row in self.frame.itertuples(index=False, name=None):
            buffer.seek(0)
            buffer.truncate(0)
            writer.writerow(row)
            self.max_row_bytes = max(self.max_row_bytes, len(buffer.getvalue().encode()))

    def partition(self, site: str) -> pd.DataFrame:
        if site not in self.site_ids:
            raise ValueError("unknown_site")
        return self.frame.iloc[self.indices[self.site_ids.index(site)]].copy()

    def identity(self) -> dict:
        return {"name": self.name, "sha256": self.sha256, "target": self.target, "source": self.source,
                "sites": self.sites, "partition_seed": self.seed, "rows": len(self.frame), "features": self.features,
                "partition_strategy": self.partition_strategy,
                "max_serialized_row_bytes_bound": self.max_row_bytes}


def load_datasets(manifest: Path) -> list[Dataset]:
    doc = yaml.safe_load(manifest.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("datasets"), list) or not doc["datasets"]:
        raise ValueError("dataset manifest must contain a nonempty datasets list")
    datasets = []
    for entry in doc["datasets"]:
        unknown = set(entry) - {"id", "path", "target", "source", "sites", "partition_seed", "partition_strategy"}
        if unknown:
            raise ValueError(f"unknown dataset manifest fields: {sorted(unknown)}")
        name = entry["id"]
        if not isinstance(name, str) or not name or not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError("dataset id must contain only letters, numbers, hyphens or underscores")
        datasets.append(Dataset(name, manifest.parent / entry["path"], entry["target"], entry["source"],
                                int(entry.get("sites", 8)), int(entry.get("partition_seed", 42)),
                                entry.get("partition_strategy", "uniform")))
    if len({d.name for d in datasets}) != len(datasets):
        raise ValueError("dataset IDs must be unique")
    return datasets


def prepare_benchmark(root: Path) -> Path:
    """Export sklearn's bundled public diabetes benchmark; no network download."""
    from sklearn.datasets import load_diabetes
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.yaml"
    if manifest.exists() or (root / "diabetes.csv").exists():
        raise FileExistsError("benchmark destination already contains a manifest or diabetes.csv")
    data = load_diabetes(as_frame=True)
    frame = data.data.copy()
    frame["target"] = data.target
    frame.to_csv(root / "diabetes.csv", index=False)
    manifest.write_text(yaml.safe_dump({"datasets": [{"id": "diabetes", "path": "diabetes.csv", "target": "target",
        "source": "scikit-learn bundled diabetes benchmark; https://scikit-learn.org/stable/datasets/toy_dataset.html",
        "sites": 8, "partition_seed": 42}]}))
    return manifest


class Trace:
    def __init__(self, run: int, group: str, user: str):
        self.run, self.group, self.user = run, group, user
        self.activities: list[dict] = []
        self.entities: list[dict] = []
        self.edges: list[dict] = []
        self._next = 1

    def entity(self, typ: str, text: str | None = None, **attributes) -> int:
        row: dict[str, Any] = dict.fromkeys(ENT_COLS)
        row.update(id=self._next, run=self.run, type=typ, text=text, digest=digest(text) if text is not None else None)
        row.update(attributes)
        self._next += 1
        self.entities.append(row)
        return int(row["id"])

    def activity(self, typ: str, start: float, end: float, used: list[int], generated: list[int],
                 informed_by: int | None = None, **attributes) -> int:
        row: dict[str, Any] = dict.fromkeys(ACT_COLS)
        row.update(id=self._next, run=self.run, type=typ, start=start, end=max(start, end),
                   seq=len(self.activities) + 1, agent="e5-agent")
        row.update(attributes)
        self._next += 1
        self.activities.append(row)
        aid = row["id"]
        for src, dst, kind in [(aid, i, "used") for i in used] + [(i, aid, "wasGeneratedBy") for i in generated]:
            self.edges.append({"kind": kind, "src": src, "dst": dst, "run": self.run})
        if informed_by is not None:
            self.edges.append({"kind": "wasInformedBy", "src": aid, "dst": informed_by, "run": self.run})
        return aid

    def graph(self) -> Graph:
        ent = pd.DataFrame(self.entities, columns=ENT_COLS)
        for c in ("n_records", "n_bytes", "out_rows", "out_bytes"):
            ent[c] = ent[c].astype(object).where(ent[c].notna(), None)
        return Graph(pd.DataFrame([{"run": self.run, "config": self.group, "user": self.user}]),
                     pd.DataFrame(self.activities, columns=ACT_COLS), ent,
                     pd.DataFrame(self.edges, columns=["kind", "src", "dst", "run"]))


def tool_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


class Tools:
    def __init__(self, dataset: Dataset, requested_sites: list[str], trace: Trace, data_seed: int,
                 workflow: str = "regression", batch_tools: bool = False):
        self.dataset, self.sites, self.trace, self.data_seed = dataset, requested_sites, trace, data_seed
        self.workflow, self.batch_tools = workflow, batch_tools
        self.profiles: dict[str, dict] = {}
        self.analysis_entities: list[int] = []
        self.described = False
        self.loaded: dict[str, pd.DataFrame] = {}
        self.loaded_entities: dict[str, int] = {}
        self.source_entities: dict[str, int] = {}
        self.validated: set[str] = set()
        self.validation_entities: dict[str, int] = {}
        self.table_entity: int | None = None
        self.table: pd.DataFrame | None = None
        self.model: Any = None
        self.model_entity: int | None = None
        self.metric_entity: int | None = None
        self.metrics: dict = {}
        self.algorithm = ""

    @property
    def finished(self) -> bool:
        return (self.described and len(self.profiles) >= min(3, len(self.dataset.features))
                if self.workflow == "explore" else bool(self.metrics))

    def next_action(self) -> str:
        pending_load = [site for site in self.sites if site not in self.loaded]
        pending_validation = [site for site in self.sites if site not in self.validated]
        if pending_load:
            tool = "load_partitions" if self.batch_tools else "load_partition"
            return f"Next: load only the pending sites with {tool}; do not repeat completed loads."
        if pending_validation:
            tool = "validate_partitions" if self.batch_tools else "validate_schema"
            return f"Next: validate only the pending sites with {tool}; do not repeat completed validations."
        if self.table is None:
            return "Next: call harmonise once; all requested sites are loaded and validated."
        if self.workflow == "explore":
            if not self.described:
                return "Next: call describe_data once."
            required = min(3, len(self.dataset.features))
            if len(self.profiles) < required:
                missing = required - len(self.profiles)
                tool = "profile_features" if self.batch_tools else "profile_feature"
                return f"Next: profile {missing} more distinct feature(s) with {tool}; avoid already profiled features."
            return "All required analysis steps are complete; provide the final report using the supplied metrics."
        if self.model is None:
            return "Next: choose an algorithm and call train_model once."
        if not self.metrics:
            return "Next: call evaluate_model once before reporting."
        return "All required analysis steps are complete; provide the final report using the supplied metrics."

    def rejection(self, name: str, arguments: Any, output: dict) -> dict:
        """Describe a rejected call and the exposed schema; the call itself is never repaired."""
        note = {"tool": name, "sent_arguments": str(arguments)[:300], "error": output.get("error", "tool_failed")}
        schema = next((s["function"]["parameters"] for s in self.schemas() if s["function"]["name"] == name), None)
        if schema is not None:
            note["expected_arguments"] = {
                k: f"JSON array of {v['minItems']}-{v['maxItems']} distinct strings from {v['items']['enum']}"
                if v.get("type") == "array" else f"one string from {v['enum']}" if "enum" in v else v.get("type", "any")
                for k, v in schema["properties"].items()}
        return note

    def state_message(self, rejected: list[dict] | None = None) -> str:
        pending_load = [site for site in self.sites if site not in self.loaded]
        pending_validation = [site for site in self.sites if site not in self.validated]
        state = {
            "requested_sites": self.sites,
            "loaded_sites": sorted(self.loaded),
            "pending_load_sites": pending_load,
            "validated_sites": sorted(self.validated),
            "pending_validation_sites": pending_validation,
            "table_ready": self.table is not None,
            "described": self.described,
            "profiled_features": sorted(self.profiles),
            "required_profiles": min(3, len(self.dataset.features)) if self.workflow == "explore" else 0,
            "model_trained": self.model is not None,
            "evaluation_complete": bool(self.metrics) if self.workflow != "explore" else False,
            "metrics": self.metrics,
        }
        message = "Current workflow state (authoritative): " + canonical(state) + "\n" + self.next_action()
        if rejected:
            # Without this the model never sees why its last call failed and repeats it verbatim.
            message += ("\nYour previous tool call(s) failed (the state above is current): " + canonical(rejected)
                        + "\nResend with arguments that match the exposed schema exactly.")
        return message

    def schemas(self, internal: bool = False) -> list[dict]:
        pending_load = [site for site in self.sites if site not in self.loaded]
        pending_validation = [site for site in self.sites if site in self.loaded and site not in self.validated]
        features_left = [feature for feature in self.dataset.features if feature not in self.profiles]
        site = {"type": "string", "enum": pending_load}
        validation_site = {"type": "string", "enum": pending_validation}
        schemas = [tool_schema("load_partition", "Load a requested CSV partition into the analysis.", {"site": site}, ["site"]),
                tool_schema("validate_schema", "Validate a loaded partition's numeric features and target.",
                            {"site": validation_site}, ["site"]),
                tool_schema("harmonise", "Combine all requested partitions after schema validation.", {}, []),
                tool_schema("describe_data", "Describe the harmonised data before training.", {}, []),
                tool_schema("train_model", "Fit a reproducible local regression model on a fixed train/test split.",
                            {"algorithm": {"type": "string", "enum": ["ridge", "linear", "random_forest"]}}, ["algorithm"]),
                tool_schema("evaluate_model", "Evaluate the fitted model on the held-out test split.", {}, [])]
        if self.workflow == "explore":
            schemas = [s for s in schemas if s["function"]["name"] not in ("train_model", "evaluate_model")]
            schemas.append(tool_schema("profile_feature", "Measure a numeric feature and its association with the target.",
                           {"feature": {"type": "string", "enum": features_left}}, ["feature"]))
        allowed = set()
        if pending_load:
            allowed.add("load_partition")
        if pending_validation:
            allowed.add("validate_schema")
        if not pending_load and not pending_validation and self.table is None:
            allowed.add("harmonise")
        if self.table is not None and not self.described:
            allowed.add("describe_data")
        if self.workflow == "explore":
            if self.described and len(self.profiles) < min(3, len(self.dataset.features)):
                allowed.add("profile_feature")
        else:
            if self.table is not None and self.model is None:
                allowed.add("train_model")
            if self.model is not None and not self.metrics:
                allowed.add("evaluate_model")
        schemas = [s for s in schemas if s["function"]["name"] in allowed]
        if self.batch_tools:
            batch_tools = []
            if pending_load:
                batch_tools.append(("load_partitions", "sites", pending_load))
            if pending_validation:
                batch_tools.append(("validate_partitions", "sites", pending_validation))
            if (self.workflow == "explore" and self.described
                    and len(self.profiles) < min(3, len(self.dataset.features))):
                batch_tools.append(("profile_features", "features", features_left))
            for name, field, values in batch_tools:
                max_items = len(values)
                if name == "profile_features":
                    max_items = min(max_items, 3 - len(self.profiles))
                schemas.append(tool_schema(name, "Execute independent per-item tools in the listed order and return results.",
                    {field: {"type": "array", "items": {"type": "string", "enum": values},
                             "minItems": 1, "maxItems": max_items, "uniqueItems": True}}, [field]))
            if not internal:
                schemas = [s for s in schemas if s["function"]["name"] not in
                           ("load_partition", "validate_schema", "profile_feature")]
        return schemas

    def _batch(self, name: str, args: Any, response_entity: int, llm_activity: int) -> tuple[dict, int]:
        pending_load = [site for site in self.sites if site not in self.loaded]
        pending_validation = [site for site in self.sites if site in self.loaded and site not in self.validated]
        features_left = [feature for feature in self.dataset.features if feature not in self.profiles]
        definitions = {"load_partitions": ("sites", pending_load, "load_partition", "site"),
                       "validate_partitions": ("sites", pending_validation, "validate_schema", "site"),
                       "profile_features": ("features", features_left, "profile_feature", "feature")}
        field, allowed, child, scalar = definitions[name]
        values = args.get(field) if isinstance(args, dict) else None
        max_items = min(3 - len(self.profiles), len(allowed)) if name == "profile_features" else len(allowed)
        valid = (isinstance(args, dict) and set(args) == {field} and isinstance(values, list)
                 and 0 < len(values) <= max_items
                 and all(isinstance(v, str) and v in allowed for v in values) and len(set(values)) == len(values))
        outputs, entities = [], []
        if valid:
            assert isinstance(values, list)
            for value in values:
                output, entity = self.execute(child, {scalar: value}, response_entity, llm_activity, internal=True)
                outputs.append(output);entities.append(entity)
            output = {"ok": all(o["ok"] for o in outputs), "results": outputs}
        else:
            output = {"ok": False, "error": "invalid_batch_arguments"}
        if not output["ok"]:
            output.update(pending_load_sites=sorted(set(self.sites) - set(self.loaded)),
                          pending_validation_sites=sorted(set(self.sites) - self.validated),
                          completed_load_sites=sorted(self.loaded),
                          completed_validation_sites=sorted(self.validated),
                          next_action=self.next_action())
        # The dispatcher really aggregates after all child executions. Each
        # independent child has its own real timestamps and explicit lineage.
        start = time.time()
        entity = self.trace.entity("batch_result" if output["ok"] else "tool_error", canonical(output))
        self.trace.activity("tool_batch", start, time.time(), [response_entity, *entities], [entity],
                            informed_by=llm_activity, tool=name, p_value=canonical(args))
        return output, entity

    def execute(self, name: str, args: Any, response_entity: int, llm_activity: int,
                internal: bool = False) -> tuple[dict, int]:
        if self.batch_tools and name in {s["function"]["name"] for s in self.schemas()} and name in (
                "load_partitions", "validate_partitions", "profile_features"):
            return self._batch(name, args, response_entity, llm_activity)
        start = time.time()
        used, generated = [response_entity], []
        try:
            schema = next((s["function"]["parameters"] for s in self.schemas(internal=internal)
                           if s["function"]["name"] == name), None)
            if schema is None:
                raise ValueError("unknown_tool")
            if not isinstance(args, dict):
                raise ValueError("invalid_tool_arguments")
            if set(args) - set(schema["properties"]) or not set(schema["required"]) <= set(args):
                raise ValueError("invalid_tool_arguments")
            for k, v in args.items():
                if not isinstance(v, str) or v not in schema["properties"][k].get("enum", [v]):
                    raise ValueError("invalid_tool_arguments")
            data, dependencies, typ = self._execute(name, args)
            used.extend(dependencies)
            output = {"ok": True, **data}
        except ValueError as exc:
            output, typ = {"ok": False, "error": str(exc),
                           "pending_load_sites": sorted(set(self.sites) - set(self.loaded)),
                           "pending_validation_sites": sorted(set(self.sites) - self.validated),
                           "completed_load_sites": sorted(self.loaded),
                           "completed_validation_sites": sorted(self.validated),
                           "next_action": self.next_action(),
                           "table_ready": self.table is not None,
                           "model_ready": self.model is not None}, "tool_error"
        entity = self.trace.entity(typ, canonical(output))
        generated.append(entity)
        if output["ok"]:
            site = str(args.get("site", ""))
            if name == "load_partition":
                self.loaded_entities[site] = entity
            elif name == "validate_schema":
                self.validation_entities[site] = entity
            elif name == "harmonise":
                self.table_entity = entity
            elif name == "train_model":
                self.model_entity = entity
            elif name == "evaluate_model":
                self.metric_entity = entity
            elif name in ("describe_data", "profile_feature") and self.workflow == "explore":
                self.analysis_entities.append(entity)
                self.metric_entity = entity
        # A rejected unknown name is a host validation event, never an executed tool.
        activity_type = name if output["ok"] else "tool_call_rejected"
        recorded_args = args if schema is not None else {"rejected_tool": name, "arguments": args}
        self.trace.activity(activity_type, start, time.time(), used, generated, informed_by=llm_activity,
                            tool=name if schema is not None else None, p_value=canonical(recorded_args),
                            site=args.get("site") if isinstance(args, dict) else None)
        return output, entity

    def _execute(self, name: str, args: dict) -> tuple[dict, list[int], str]:
        ds = self.dataset
        site = str(args.get("site", ""))
        if name == "load_partition":
            if site in self.loaded:
                raise ValueError("partition_already_loaded")
            frame = ds.partition(site)
            csv = frame.to_csv(index=False).encode()
            src = self.trace.entity("data_partition", digest=digest(csv), site=site,
                                    part_id=f"{ds.name}/{ds.sha256}/{ds.partition_strategy}/{ds.seed}/{site}",
                                    n_records=len(frame), n_bytes=len(csv))
            self.loaded[site], self.source_entities[site] = frame, src
            return {"site": site, "rows": len(frame), "columns": list(frame.columns)}, [src], "loaded_partition"
        if name == "validate_schema":
            if site not in self.loaded:
                raise ValueError("load_partition_first")
            self.validated.add(site)
            return {"site": site, "valid": True, "target": ds.target}, [self.loaded_entities[site]], "schema_validation"
        if name == "harmonise":
            if self.validated != set(self.sites):
                raise ValueError("load_and_validate_all_requested_sites_first")
            self.table = pd.concat([self.loaded[s] for s in self.sites], ignore_index=True)
            return {"rows": len(self.table), "features": ds.features, "target": ds.target}, \
                [self.loaded_entities[s] for s in self.sites] + [self.validation_entities[s] for s in self.sites], "table"
        if self.table is None or self.table_entity is None:
            raise ValueError("harmonise_first")
        if name == "describe_data":
            self.described = True
            if self.workflow == "explore":
                self.metrics = {"workflow": "explore", "rows": len(self.table),
                                "profiles": self.profiles, "target_mean": float(self.table[ds.target].mean())}
            return {"rows": len(self.table), "target_mean": float(self.table[ds.target].mean()),
                    "feature_missing": self.table[ds.features].isna().sum().astype(int).to_dict()}, [self.table_entity], "summary"
        if name == "profile_feature":
            if not self.described:
                raise ValueError("describe_data_first")
            feature = args["feature"]
            series = self.table[feature]
            corr = series.corr(self.table[ds.target])
            profile = {"feature": feature, "count": int(series.count()), "missing": int(series.isna().sum()),
                       "mean": float(series.mean()), "std": float(series.std()), "min": float(series.min()),
                       "max": float(series.max()), "target_correlation": float(corr) if np.isfinite(corr) else None}
            self.profiles[feature] = profile
            if self.described:
                self.metrics["profiles"] = self.profiles
            return profile, [self.table_entity], "feature_profile"
        if name == "train_model":
            self.algorithm = args["algorithm"]
            models = {"ridge": Ridge(alpha=1.0), "linear": LinearRegression(),
                      "random_forest": RandomForestRegressor(n_estimators=30, max_depth=5, n_jobs=1,
                                                              random_state=self.data_seed)}
            xtr, self.xtest, ytr, self.ytest = train_test_split(self.table[ds.features], self.table[ds.target],
                                                                             test_size=0.25, random_state=self.data_seed)
            self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), models[self.algorithm])
            self.model.fit(xtr, ytr)
            self.metric_entity = None
            self.metrics = {}
            return {"algorithm": self.algorithm, "train_rows": len(xtr), "test_rows": len(self.xtest)}, \
                [self.table_entity], "fitted_model"
        if name == "evaluate_model":
            if self.model is None or self.model_entity is None:
                raise ValueError("train_model_first")
            predicted = self.model.predict(self.xtest)
            self.metrics = {"algorithm": self.algorithm, "mse": float(mean_squared_error(self.ytest, predicted)),
                            "mae": float(mean_absolute_error(self.ytest, predicted)),
                            "r2": float(r2_score(self.ytest, predicted)), "test_rows": len(self.ytest)}
            return self.metrics, [self.table_entity, self.model_entity], "evaluation"
        raise ValueError("unknown_tool")


ANALYST_PROFILES = {
    "profile-0": "Prefer a concise analysis and a brief final report.",
    "profile-1": "Emphasise missing values and schema quality in your final report.",
    "profile-2": "Emphasise interpretation of measured numeric results and their limitations.",
    "profile-3": "Prefer a robust analysis; explain your methodological choice briefly.",
    "profile-4": "Prefer a simple interpretable analysis; clearly name the variables examined.",
    "profile-5": "Emphasise reproducibility and state which computations support your conclusions.",
}


def initial_messages(dataset: Dataset, sites: list[str], data_seed: int = 42, workflow: str = "regression",
                     batch_tools: bool = False, profile: str | None = None) -> list[dict]:
    task = ("exploratory multi-site data analysis" if workflow == "explore" else "multi-site tabular regression")
    required_profiles = min(3, len(dataset.features))
    goal = (f"Describe missingness and profile at least {required_profiles} distinct numeric feature(s), "
            "then give a concise report. "
            "Do not fit a predictive model. " if workflow == "explore" else
            "Inspect the data if useful, choose a model, train it, evaluate the held-out split, then report. "
            "Do not finish before evaluate_model succeeds. ")
    bulk = ("Follow the exposed phases in order: call load_partitions once for all requested sites, then "
            "validate_partitions once for all pending sites, then harmonise once. For exploration, call "
            f"describe_data once and profile at least {required_profiles} distinct feature(s) with profile_features. "
            "Each batch tool records its independent child operations. Never repeat a successful phase. "
            if batch_tools else "")
    style = ANALYST_PROFILES[profile] if profile else ""
    return [{"role": "system", "content": "You are a data-analysis agent. Use only the supplied tools. "
             "Load and validate every requested site, then harmonise the data. " + bulk + goal +
             "Never invent metrics. The current-state message and exposed tools define the remaining work. "
             "A successful phase must not be repeated. Complete every required phase before reporting. "
             "Correct tool errors using the pending-site feedback. " + style},
            {"role": "user", "content": canonical({"task": task, "dataset": dataset.name,
             "target": dataset.target, "requested_sites": sites, "split_seed": data_seed})}]


def make_spec(model_id: str, temperature: float | None, workflow: str = "regression", batch_tools: bool = False) -> Spec:
    names = ["load_partition", "validate_schema", "harmonise", "describe_data", "train_model", "evaluate_model"]
    if workflow == "explore":
        names = ["load_partition", "validate_schema", "harmonise", "describe_data", "profile_feature"]
    batches = (["load_partitions", "validate_partitions"] + (["profile_features"] if workflow == "explore" else [])
               if batch_tools else [])
    cons = [Constraint("init", "agent_step"), Constraint("existence", "agent_report")]
    required = (names if workflow == "explore" else [n for n in names if n != "describe_data"])
    cons += [Constraint("existence", name) for name in required]
    chain = (["load_partition", "validate_schema", "harmonise", "describe_data", "profile_feature", "agent_report"]
             if workflow == "explore" else
             ["load_partition", "validate_schema", "harmonise", "train_model", "evaluate_model", "agent_report"])
    cons += [Constraint("precedence", a, b) for a, b in zip(chain, chain[1:])]
    return Spec(name="E5 captured API agent with local CSV tools",
                alphabet=frozenset(names + ["agent_step", "agent_report", "tool_call_rejected", "tool_batch"]),
                constraints=cons, vocab={"tool": frozenset(names + batches), "model": frozenset({model_id}),
                                        "temperature": frozenset({str(temperature) if temperature is not None
                                                                  else "provider-default"})},
                vocab_rules=[*[VocabRule(n, "tool", frozenset({n})) for n in names],
                             *[VocabRule(n, "model", frozenset({model_id})) for n in ("agent_step", "agent_report")],
                             *([VocabRule("tool_batch", "tool", frozenset(batches))] if batches else [])],
                generalisers={}, nd_types=frozenset({"agent_step", "agent_report"}), nd_requires_seed=False,
                thresholds=[ThresholdRule("data_partition", "n_records", 12)], timed=[TimedRule("span", 3600)])


def run_agent(dataset: Dataset, sites: list[str], client, model_id: str, temperature: float | None,
              run: int, group: str, capture_dir: Path, data_seed: int = 42, max_steps: int = 16,
              workflow: str = "regression", batch_tools: bool = False, profile: str | None = None) -> tuple[Graph, dict]:
    trace = Trace(run, group, profile or "benchmark-analyst")
    tools = Tools(dataset, sites, trace, data_seed, workflow, batch_tools)
    base_messages = initial_messages(dataset, sites, data_seed, workflow, batch_tools, profile)
    messages = base_messages.copy()
    captures = []
    journal = capture_dir / f"run_{run:05d}.jsonl"
    for step in range(max_steps):
        prompt = canonical(messages)
        prompt_entity = trace.entity("prompt", prompt)
        start = time.time()
        completion: Completion = client.complete(messages, tools.schemas())
        response_entity = trace.entity("response", canonical(completion.message))
        calls = completion.message.get("tool_calls", [])
        record = {"step": step, "messages": messages.copy(), "response": completion.response,
                  "request_without_credentials": completion.request,
                  "message": completion.message, "input_tokens": completion.input_tokens,
                  "output_tokens": completion.output_tokens, "attempts": completion.attempts,
                  "mode": "replay" if completion.replay else "mock" if completion.mock else "live"}
        captures.append(record)
        with journal.open("a", encoding="utf-8") as f:
            f.write(canonical(record) + "\n")
            f.flush()
        activity = trace.activity("agent_report" if not calls and tools.finished else "agent_step", start, time.time(),
                                  [prompt_entity], [response_entity], model=model_id,
                                  temperature=str(temperature) if temperature is not None else "provider-default", seed=None)
        if not calls:
            if not tools.finished or tools.metric_entity is None:
                messages = base_messages + [{"role": "user", "content": tools.state_message()}]
                continue
            result = trace.entity("result", canonical({"metrics": tools.metrics, "report": completion.message["content"]}))
            trace.edges.append({"kind": "wasGeneratedBy", "src": result, "dst": activity, "run": run})
            for entity in tools.analysis_entities if workflow == "explore" else [tools.metric_entity]:
                trace.edges.append({"kind": "used", "src": activity, "dst": entity, "run": run})
            atomic_json(capture_dir / f"run_{run:05d}.json", captures)
            return trace.graph(), {"steps": step + 1, "model_calls": sum(c["mode"] == "live" for c in captures),
                                  "metrics": tools.metrics, "sites": sites, "captures": captures}
        rejected = []
        for call in calls:
            try:
                args = json.loads(call["function"]["arguments"])
            except (TypeError, ValueError):
                # Preserve the malformed string. The tool wrapper rejects it,
                # records the error and returns current pending work to the model.
                args = call["function"]["arguments"]
            output, _ = tools.execute(call["function"]["name"], args, response_entity, activity)
            if not output["ok"]:
                rejected.append(tools.rejection(call["function"]["name"], call["function"]["arguments"], output))
        # Each request starts from the task, the deterministic current state and
        # only the last turn's rejections, so the prompt cannot grow without bound.
        messages = base_messages + [{"role": "user", "content": tools.state_message(rejected)}]
    raise ProviderError("agent_step_limit_exceeded")


class ReplayClient:
    """Replay captured decisions, re-execute tools; never claim fresh LLM inference."""
    def __init__(self, captures: list[dict]):
        self.captures, self.index = captures, 0

    def complete(self, messages, tools) -> Completion:
        if self.index >= len(self.captures):
            raise ProviderError("replay_exhausted")
        row = self.captures[self.index]
        # Exact prompt equality checks that re-executed tool outputs match the capture.
        if canonical(messages) != canonical(row["messages"]):
            raise ProviderError("replay_prompt_or_tool_output_mismatch")
        self.index += 1
        return Completion(row["message"], row["response"], attempts=0, replay=True)


class MockClient:
    """Offline protocol fixture. Its output is always labelled mock/smoke."""
    def __init__(self):
        self.step = 0

    def complete(self, messages, tools) -> Completion:
        available = {t["function"]["name"]: t["function"] for t in tools}
        name = None
        args: dict = {}
        if "load_partitions" in available:
            name = "load_partitions"
            args = {"sites": available[name]["parameters"]["properties"]["sites"]["items"]["enum"]}
        elif "load_partition" in available:
            name = "load_partition"
            args = {"site": available[name]["parameters"]["properties"]["site"]["enum"][0]}
        elif "validate_partitions" in available:
            name = "validate_partitions"
            args = {"sites": available[name]["parameters"]["properties"]["sites"]["items"]["enum"]}
        elif "validate_schema" in available:
            name = "validate_schema"
            args = {"site": available[name]["parameters"]["properties"]["site"]["enum"][0]}
        elif "harmonise" in available:
            name = "harmonise"
        elif "describe_data" in available:
            name = "describe_data"
        elif "profile_features" in available:
            name = "profile_features"
            enum = available[name]["parameters"]["properties"]["features"]["items"]["enum"]
            args = {"features": enum[:3]}
        elif "profile_feature" in available:
            name = "profile_feature"
            args = {"feature": available[name]["parameters"]["properties"]["feature"]["enum"][0]}
        elif "train_model" in available:
            name, args = "train_model", {"algorithm": "ridge"}
        elif "evaluate_model" in available:
            name = "evaluate_model"
        if name:
            message = {"role": "assistant", "content": "", "tool_calls": [{"id": f"mock-{self.step}", "type": "function",
                       "function": {"name": name, "arguments": canonical(args)}}]}
        else:
            message = {"role": "assistant", "content": "Offline fixture completed; see captured evaluation metrics."}
        self.step += 1
        return Completion(message, {"model": "MOCK-NOT-A-LIVE-LLM", "choices": [{"message": message}]}, mock=True)
