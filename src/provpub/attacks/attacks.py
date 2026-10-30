"""Inference attacks by an honest-but-curious verifier.

The verifier reads only the published release, the public specification,
the operator configuration (Kerckhoffs: mechanism known, key unknown), any
public salt, and the auxiliary knowledge stated per attack.

A1 site membership     : did target site s* contribute data to run r?
A2 size class          : small / medium / large class of each data partition
A3 analyst re-identification from timing
A4 dictionary attack on prompt pseudonyms
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from ..data.schemas import SUPP, Graph
from ..data.synthetic import size_class_of
from ..methods.operators import PS, salted_pseudonym


def _num(v):
    if v is None or (isinstance(v, str)):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


class _Clf:
    """Gradient boosting on the informative columns; majority vote if none."""

    def __init__(self, seed=0, iters=150):
        self.seed, self.iters = seed, iters

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=-1.0, posinf=-1.0, neginf=-1.0)
        self.cols = [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) > 1]
        self.classes = np.unique(y)
        self.major = np.bincount(y).argmax()
        if self.cols:
            self.m = HistGradientBoostingClassifier(max_iter=self.iters, random_state=self.seed)
            self.m.fit(X[:, self.cols], y)
        return self

    def predict_proba(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=-1.0, posinf=-1.0, neginf=-1.0)
        if not self.cols:
            p = np.zeros((len(X), len(self.classes)))
            p[:, list(self.classes).index(self.major)] = 1.0
            return p
        return self.m.predict_proba(X[:, self.cols])

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]


def _salted_ps(salt: bytes, attr: str, value: str, nbytes: int = 8) -> str:
    return salted_pseudonym(salt, attr, value, nbytes)


def _run_text(h: Graph) -> dict:
    """All clear-text string values published for each run (prompts, responses,
    identifiers and other attributes), for name matching."""
    out = defaultdict(list)
    for tab in (h.act, h.ent):
        for c in tab.columns:
            if c in ("id", "run", "type", "digest") or pd.api.types.is_numeric_dtype(tab[c]):
                continue
            for r, v in zip(tab["run"].to_numpy(), tab[c].to_numpy(dtype=object)):
                if isinstance(v, str) and v != SUPP and not v.startswith(PS):
                    out[r].append(v)
    for c in h.runs.columns:
        if c == "run" or pd.api.types.is_numeric_dtype(h.runs[c]):
            continue
        for r, v in zip(h.runs["run"].to_numpy(), h.runs[c].to_numpy(dtype=object)):
            if isinstance(v, str) and v != SUPP and not v.startswith(PS):
                out[r].append(v)
    return {r: "\n".join(v) for r, v in out.items()}


def partition_table(h: Graph, spec) -> pd.DataFrame:
    """One row per published data partition with the fields a verifier sees."""
    ent, act, e = h.ent, h.act, h.edge
    src = ent[ent["type"] == spec.source_type][["id", "run", "digest", "site", "n_records", "n_bytes"]].copy()
    use = e[e["kind"] == "used"][["src", "dst"]].rename(columns={"src": "act", "dst": "id"})
    a = act[act["type"] == "load_partition"].copy()
    if "b_start" in a.columns:
        a["dur"] = (a["b_end"] - a["b_start"]).astype(float) * float(h.meta.get("delta", 1.0))
        a["pos"] = a["rank"]
    elif "start" in a.columns and pd.api.types.is_numeric_dtype(a["start"]):
        a["dur"] = (a["end"] - a["start"]).astype(float)
        a["pos"] = a.sort_values(["run", "start", "seq"]).groupby("run").cumcount().reindex(a.index)
    else:
        a["dur"] = np.nan
        a["pos"] = np.nan
    a = a[["id", "dur", "pos", "site"]].rename(columns={"id": "act", "site": "act_site"})
    t = src.merge(use, on="id", how="left").merge(a, on="act", how="left")
    # adversary post-processes unclamped noisy values to be non-negative
    t["n"] = t["n_records"].map(_num).clip(lower=0)
    t["b"] = t["n_bytes"].map(_num).clip(lower=0)
    t["w"] = t["b"] / t["n"].where(t["n"] > 0)
    # load-order position relative to the first load in the run
    t["pos"] = t["pos"] - t.groupby("run")["pos"].transform("min")
    return t


# ------------------------------------------------------------------ A1

def membership_attack(h: Graph, spec, truth: dict, sites, targets, known_frac=0.2, seed=0,
                      salt: bytes | None = None, aux_noise=0.1):
    """Returns (mean AUC over targets on unknown runs, per-target AUCs,
    mean AUC restricted to runs with no input-aggregate link to a labelled run).

    Auxiliary knowledge: the public list of consortium sites; for each target,
    its approximate record count, row width and load duration (each within
    +/- aux_noise); full membership labels for a random known_frac of runs.
    Linkage features of labelled runs are computed leave-one-out. The run map
    is used only to attach ground truth, never as a feature.
    """
    from collections import Counter
    rng = np.random.default_rng(seed)
    inv = h.meta.get("run_map_inv", {})
    runs = h.runs["run"].to_numpy()
    orig = np.array([inv.get(int(r), int(r)) for r in runs])
    known = rng.random(len(runs)) < known_frac
    P = partition_table(h, spec)
    ridx = {r: i for i, r in enumerate(runs)}
    P = P[P["run"].isin(ridx)].copy()
    P["ri"] = P["run"].map(ridx)
    P = P.sort_values("ri", kind="mergesort").reset_index(drop=True)
    ri = P["ri"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ri[1:] != ri[:-1]]) if len(ri) else np.array([], dtype=int)
    has = np.zeros(len(runs), dtype=bool)
    has[ri[starts]] = True
    k_parts = np.zeros(len(runs))
    k_parts[ri[starts]] = np.diff(np.r_[starts, len(ri)])
    n = P["n"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lw = np.log(np.clip(P["w"].to_numpy(dtype=float), 1e-9, None))
    dur = P["dur"].to_numpy(dtype=float)
    tok = P["digest"].astype(str).to_numpy()
    # published (count, bytes) pair as a value token: identical exact or memoised
    # noisy values link occurrences of one partition across runs
    nb = P["b"].to_numpy(dtype=float)
    vtok = np.array([f"{a:.0f}|{b:.0f}" if np.isfinite(a) and np.isfinite(b) else SUPP for a, b in zip(n, nb)], dtype=object)
    # averaging attack: mean published count over occurrences sharing a pseudonym
    tser = pd.Series(n)
    n_avg = tser.groupby(tok).transform("mean").to_numpy().copy()
    n_avg[tok == SUPP] = n[tok == SUPP]
    names = np.array([f"{a}|{b}" for a, b in zip(P["site"].tolist(), P["act_site"].tolist())], dtype=object)
    in_dig = dict(zip(h.runs["run"], h.runs["input_digest"])) if "input_digest" in h.runs.columns else {}
    run_text = _run_text(h)
    run_in = np.array([in_dig.get(r) for r in runs], dtype=object)

    def per_run_min(v):
        out = np.full(len(runs), np.nan)
        if len(v) == 0:
            return out
        vv = np.where(np.isfinite(v), v, np.inf)
        m = np.minimum.reduceat(vv, starts)
        m[~np.isfinite(m)] = np.nan
        out[ri[starts]] = m
        return out

    aucs, aucs_fresh = {}, {}
    for si in targets:
        s = sites[si]
        n_aux = s.base_records * (1 + rng.uniform(-aux_noise, aux_noise))
        w_aux = s.row_width * (1 + rng.uniform(-aux_noise, aux_noise))
        d_aux = (s.latency + s.base_records / s.throughput) * (1 + rng.uniform(-aux_noise, aux_noise))
        y = np.array([int(s.name in truth[o]["sites"]) for o in orig])
        nm = {s.name}
        if salt is not None:
            nm.add(_salted_ps(salt, "site", s.name))
        hit = np.array([any(x in nm for x in v.split("|")) for v in names], dtype=float)
        f_name = per_run_min(-hit)
        f_name = np.where(np.isnan(f_name), 0.0, -f_name)
        # clear-text mention of the site name anywhere in the run's published strings
        pat = re.compile(r"(?<![\w-])" + re.escape(s.name) + r"(?![\w-])")
        f_text = np.array([1.0 if pat.search(run_text.get(r, "")) else 0.0 for r in runs])
        f_name = np.maximum(f_name, f_text)
        ds = np.abs(np.log1p(n) - math.log(n_aux))
        dsa = np.abs(np.log1p(n_avg) - math.log(n_aux))
        dw = np.abs(lw - math.log(w_aux))
        dd = np.abs(np.log1p(dur) - math.log1p(d_aux))
        comb = np.nan_to_num(ds, nan=0) + np.nan_to_num(dw, nan=0) + np.nan_to_num(dd, nan=0)
        informative = np.isfinite(ds) | np.isfinite(dd)
        comb = np.where(informative, comb, np.nan)
        f_size, f_w, f_d, f_comb = per_run_min(ds), per_run_min(dw), per_run_min(dd), per_run_min(comb)
        f_avg = per_run_min(dsa)
        # best-matching partition token of each labelled positive run
        sc = np.nan_to_num(ds, nan=0) + np.nan_to_num(dd, nan=0)
        sc = np.where(informative & (tok != SUPP), sc, np.inf)
        sc_v = np.where(informative & (vtok != SUPP), np.nan_to_num(ds, nan=0) + np.nan_to_num(dd, nan=0), np.inf)

        def linkage(tokens, score):
            best = {}
            if len(score):
                mins = np.minimum.reduceat(score, starts)
                for j, st in enumerate(starts):
                    if np.isfinite(mins[j]):
                        en = starts[j + 1] if j + 1 < len(starts) else len(score)
                        best[ri[st]] = tokens[st + int(np.argmin(score[st:en]))]
            pos = Counter(best[i] for i in np.flatnonzero(known) if y[i] and i in best)  # noqa: B023 (called inside the loop iteration)
            out = np.zeros(len(runs))
            for j, st in enumerate(starts):
                i = ri[st]
                en = starts[j + 1] if j + 1 < len(starts) else len(tokens)
                cnt = {t: pos[t] for t in set(tokens[st:en]) if t in pos}
                if known[i] and y[i] and best.get(i) in cnt:  # noqa: B023 (called inside the loop iteration)
                    cnt[best[i]] -= 1
                out[i] = float(any(v > 0 for v in cnt.values()))
            return out

        link_tok = linkage(tok, sc)
        link_val = linkage(vtok, sc_v)
        pos_in: Counter = Counter()
        neg_in: Counter = Counter()
        for i in np.flatnonzero(known):
            if run_in[i] is not None:
                (pos_in if y[i] else neg_in)[run_in[i]] += 1
        link_in = np.full(len(runs), 0.5)
        for run_index in range(len(runs)):
            d = run_in[run_index]
            if d is None:
                continue
            p_, n_ = pos_in[d], neg_in[d]
            if known[run_index]:
                p_ -= int(y[run_index] == 1)
                n_ -= int(y[run_index] == 0)
            if p_ > 0 and n_ == 0:
                link_in[run_index] = 1.0
            elif n_ > 0 and p_ == 0:
                link_in[run_index] = 0.0
        X = np.column_stack([f_name, f_size, f_w, f_d, f_comb, f_avg, k_parts, link_tok, link_val, link_in])
        tr, te = known, ~known
        if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
            continue
        clf = _Clf(seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs[s.name] = roc_auc_score(y[te], p)
        fresh = link_in[te] == 0.5
        if fresh.sum() > 10 and 0 < y[te][fresh].sum() < fresh.sum():
            aucs_fresh[s.name] = roc_auc_score(y[te][fresh], p[fresh])
    mean = lambda d: float(np.mean(list(d.values()))) if d else float("nan")
    return mean(aucs), aucs, mean(aucs_fresh)


# ------------------------------------------------------------------ A2

def _aia_features(h: Graph, spec):
    P = partition_table(h, spec)
    # averaging across occurrences that share a published token (helps against fresh noise)
    tok = P["digest"].astype(str)
    grp = P.groupby(tok)
    P["n_mean"] = grp["n"].transform("mean")
    P["b_mean"] = grp["b"].transform("mean")
    P["occ"] = grp["n"].transform("size").astype(float)
    P.loc[tok == SUPP, ["n_mean", "b_mean", "occ"]] = np.nan
    X = np.column_stack([np.log1p(P["n"]), np.log1p(P["b"]), np.log1p(P["n_mean"]), np.log1p(P["b_mean"]),
                         np.log1p(P["dur"]), P["occ"]])
    return P, X


def size_class_attack(h_target: Graph, h_shadow: Graph, spec):
    """Train on a shadow release that the adversary produced with the same
    mechanism (its own data and key) and test on the target release.
    Both releases carry publisher-side ground truth (attach_truth) used only
    for training labels (shadow) and scoring (target)."""
    _, Xs = _aia_features(h_shadow, spec)
    _, Xt = _aia_features(h_target, spec)
    ys = _true_classes(h_shadow, spec)
    yt = _true_classes(h_target, spec)
    clf = _Clf(0, 200)
    clf.fit(Xs, ys)
    pred = clf.predict(Xt)
    acc = accuracy_score(yt, pred)
    base = np.bincount(yt).max(axis=0) / len(yt)
    return acc, base


def _true_classes(h: Graph, spec):
    """Publisher-side ground truth: true size class of each published partition,
    aligned with partition_table(h) row order via the private node mapping."""
    P = partition_table(h, spec)
    m = h.meta.get("node_truth")
    if m is None:
        raise RuntimeError("release lacks node_truth; build with attach_truth()")
    return np.array([size_class_of(m[int(i)]) for i in P["id"]])


def attach_truth(raw: Graph, h: Graph, ring_key: bytes, spec):
    """Publisher-side: map published partition ids to true record counts (for scoring only)."""
    import hmac as _h
    src = raw.ent[raw.ent["type"] == spec.source_type]
    renum = "VG" in h.meta.get("ops", [])
    m = {}
    for i, n in zip(src["id"].to_numpy(), src["n_records"].to_numpy()):
        if renum:
            nid = int.from_bytes(_h.digest(ring_key, f"nid|{i}".encode(), "sha256")[:7], "big") >> 4
        else:
            nid = int(i)
        m[nid] = int(n)
    # abstraction replaces partitions by a pooled entity: label it by the run's largest partition
    if "ABSTRACT" in h.meta.get("ops", []):
        P = partition_table(h, spec)
        big = src.groupby("run")["n_records"].max().to_dict()
        for i, r in zip(P["id"], P["run"]):
            m.setdefault(int(i), int(big.get(r, 0)))
    h.meta["node_truth"] = m
    return h


# ------------------------------------------------------------------ A3

def analyst_attack(h: Graph, truth: dict, analysts, known_frac=0.2, seed=0, salt: bytes | None = None):
    rng = np.random.default_rng(seed)
    inv = h.meta.get("run_map_inv", {})
    runs = h.runs["run"].to_numpy()
    orig = np.array([inv.get(int(r), int(r)) for r in runs])
    names = [a.name for a in analysts]
    y = np.array([names.index(truth[o]["user"]) for o in orig])
    act = h.act
    if "b_start" in act.columns:
        d = float(h.meta["delta"])
        s = act["b_start"].astype(float) * d
        e = act["b_end"].astype(float) * d
        hour = pd.Series(np.nan, index=h.runs["run"])
    elif "start" in act.columns and pd.api.types.is_numeric_dtype(act["start"]):
        s, e = act["start"].astype(float), act["end"].astype(float)
        hour = (act.groupby("run")["start"].min() % 86400) / 3600.0
    else:
        s = e = None
        hour = pd.Series(np.nan, index=h.runs["run"])
    feats = pd.DataFrame(index=runs)
    feats["hour"] = hour.reindex(runs).to_numpy()
    if s is not None:
        df = pd.DataFrame({"run": act["run"], "type": act["type"], "s": s, "e": e})
        reason_end = df[df["type"] == "agent_reason"].groupby("run")["e"].max()
        train_start = df[df["type"] == "train_model"].groupby("run")["s"].min()
        feats["wait"] = (train_start - reason_end).reindex(runs).to_numpy()
        feats["span"] = (df.groupby("run")["e"].max() - df.groupby("run")["s"].min()).reindex(runs).to_numpy()
    else:
        feats["wait"] = np.nan
        feats["span"] = np.nan
    if "date" in h.runs.columns and pd.api.types.is_numeric_dtype(h.runs["date"]):
        feats["date"] = h.runs.set_index("run")["date"].reindex(runs).to_numpy(dtype=float)
    else:
        feats["date"] = np.nan
    X = feats.to_numpy(dtype=float)
    known = rng.random(len(runs)) < known_frac
    # direct read of the user field (clear text, or salted hash of a staff list)
    users = h.runs["user"].astype(str).to_numpy()
    direct = np.full(len(runs), -1)
    for k, nm in enumerate(names):
        direct[users == nm] = k
        if salt is not None:
            direct[users == _salted_ps(salt, "user", nm)] = k
    clf = _Clf(seed)
    clf.fit(X[known], y[known])
    pred = clf.predict(X[~known])
    dk = direct[~known]
    pred = np.where(dk >= 0, dk, pred)
    acc = accuracy_score(y[~known], pred)
    base = np.bincount(y[known], minlength=len(names)).argmax(axis=0)
    return acc, float(np.mean(y[~known] == base))


# ------------------------------------------------------------------ A4

def dictionary_attack(h: Graph, candidates: list[str], salt: bytes | None):
    """Fraction of distinct published prompt pseudonyms recovered by
    enumerating the candidate prompt space. Without the key (keyed HMAC)
    the attacker cannot evaluate the pseudonym function; it can only test
    candidates against values published in clear."""
    ent = h.ent
    pub = set(ent[ent["type"] == "prompt"]["text"].astype(str))
    if salt is None:
        table = set(candidates)            # only clear-text matches are possible
    else:
        table = {_salted_ps(salt, "text", c) for c in candidates}
    hit = len(pub & table)
    return hit / max(1, len(pub)), len(pub)


# ------------------------------------------------------------------ A5

def position_attack(h: Graph, raw: Graph, spec, key: bytes):
    """Collection-order side channel. The adversary assumes that partitions are
    loaded in alphabetical order of site names (the configuration order of the
    generator) and guesses that the j-th published load of a run belongs to the
    site with the j-th smallest name among the run's sites. Scored per load
    activity against publisher-side truth; chance is the mean of 1/k over loads.
    Returns (accuracy, chance) or (nan, nan) if positions are not published."""
    import hmac as _h
    act = h.act
    if "rank" in act.columns:
        pos = act["rank"]
    elif "start" in act.columns and pd.api.types.is_numeric_dtype(act["start"]) and "seq" in act.columns:
        pos = act.sort_values(["run", "start", "seq"]).groupby("run").cumcount().reindex(act.index)
    else:
        return float("nan"), float("nan")
    ra = raw.act[raw.act["type"] == "load_partition"]
    renum = "VG" in h.meta.get("ops", [])
    true_site = {}
    for i, sname in zip(ra["id"].to_numpy(), ra["site"].to_numpy(dtype=object)):
        nid = (int.from_bytes(_h.digest(key, f"nid|{i}".encode(), "sha256")[:7], "big") >> 4) if renum else int(i)
        true_site[nid] = sname
    la = act[act["type"] == "load_partition"].assign(pos=pos)
    la = la[la["id"].notna()]
    if la.empty:
        return float("nan"), float("nan")
    la = la[la["id"].astype("int64").isin(true_site)]
    if la.empty or la["pos"].isna().all():
        return float("nan"), float("nan")
    la = la.assign(site=[true_site[int(i)] for i in la["id"]])
    hits, n, chance = 0, 0, 0.0
    for _, grp in la.groupby("run"):
        order = grp.sort_values("pos", kind="mergesort")["site"].tolist()
        hits += sum(a == b for a, b in zip(order, sorted(order)))
        n += len(order)
        chance += 1.0  # each of the k loads is correct with probability 1/k under a random order
    return hits / n, chance / n
