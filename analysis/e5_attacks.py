"""Verifier-side E5 attacks with task/source-cluster uncertainty.

Only features observable in a release enter models. Publisher maps are used
solely for attaching labels/scoring. Replays never enter attack training or
confidence intervals. A3 concerns six simulated analyst profiles, not people.
"""

from __future__ import annotations

import hmac
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from provpub.attacks.attacks import partition_table
from provpub.e5.workflow import ANALYST_PROFILES, canonical, initial_messages
from provpub.methods.operators import salted_pseudonym

SEEDS = (0, 1, 2, 3, 4)
KNOWN = 0.25
BOOTSTRAPS = 1000


def model(seed=0):
    # Fixed before collection: small labelled samples per public dataset.
    return HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, max_leaf_nodes=7, min_samples_leaf=2, l2_regularization=1.0, random_state=seed
    )


def matrix(frame):
    return np.nan_to_num(np.asarray(frame, dtype=float), nan=-1.0, posinf=-1.0, neginf=-1.0)


def original_runs(h):
    inv = h.meta.get("run_map_inv", {})
    return {int(r): int(inv.get(int(r), int(r))) for r in h.runs["run"]}


def node_map(raw, h, key):
    hashed = "VG" in h.meta.get("ops", [])
    return {
        int(i): (int.from_bytes(hmac.digest(key, f"nid|{int(i)}".encode(), "sha256")[:7], "big") >> 4 if hashed else int(i))
        for i in pd.concat([raw.act["id"], raw.ent["id"]])
    }


def known_indices(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    tr = order[: max(2, int(round(n * KNOWN)))]
    te = order[len(tr) :]
    return tr, te


def bootstrap_counts(frame, cluster="task", seed=20261026):
    clusters = frame[["dataset", cluster]].drop_duplicates().reset_index(drop=True)
    # Cluster identifiers only need to be unique within a dataset. Keep the
    # dataset in the key so equally named tasks from different cohorts are
    # resampled independently rather than silently sharing bootstrap weights.
    lookup = {
        (str(dataset), str(value)): i
        for i, (dataset, value) in enumerate(clusters[["dataset", cluster]].itertuples(index=False, name=None))
    }
    counts = np.zeros((len(clusters), BOOTSTRAPS))
    rng = np.random.default_rng(seed)
    for _, group in clusters.groupby("dataset"):
        ids = group.index.to_numpy()
        n = len(ids)
        counts[ids] = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAPS).T
    return counts, lookup


def accuracy_interval(frame, value="correct", cluster="task"):
    if frame.empty:
        return {"estimate": None, "ci_low": None, "ci_high": None, "clusters": 0}
    # Average repeated occurrences/splits within the declared cluster first.
    scores = frame.groupby(["dataset", cluster], as_index=False)[value].mean()
    counts, lookup = bootstrap_counts(scores, cluster)
    keys = [(str(dataset), str(task)) for dataset, task in zip(scores["dataset"], scores[cluster])]
    weights = counts[[lookup[key] for key in keys]]
    b = (weights * scores[value].to_numpy()[:, None]).sum(axis=0) / weights.sum(axis=0)
    return {
        "estimate": float(scores[value].mean()),
        "ci_low": float(np.quantile(b, 0.025)),
        "ci_high": float(np.quantile(b, 0.975)),
        "clusters": len(scores),
    }


def auc_interval(frame):
    if frame.empty:
        return {"estimate": None, "ci_low": None, "ci_high": None, "clusters": 0}
    counts, lookup = bootstrap_counts(frame)
    estimates, draws = [], []
    for _, group in frame.groupby(["dataset", "target", "split"]):
        if group["y"].nunique() != 2:
            continue
        estimates.append(roc_auc_score(group.y, group.score))
        group = group.sort_values("score", kind="stable")
        y = group.y.to_numpy(dtype=float)
        dataset = str(group.dataset.iloc[0])
        w = counts[[lookup[(dataset, str(task))] for task in group.task]]
        starts = np.r_[0, np.flatnonzero(np.diff(group.score.to_numpy())) + 1]
        pos = np.add.reduceat(w * y[:, None], starts, axis=0)
        neg = np.add.reduceat(w * (1 - y[:, None]), starts, axis=0)
        numerator = (pos * (np.cumsum(neg, axis=0) - 0.5 * neg)).sum(axis=0)
        denominator = pos.sum(axis=0) * neg.sum(axis=0)
        draws.append(np.divide(numerator, denominator, out=np.full(BOOTSTRAPS, np.nan), where=denominator > 0))
    if not estimates:
        return {"estimate": None, "ci_low": None, "ci_high": None, "clusters": len(lookup)}
    values = np.nanmean(draws, axis=0)
    return {
        "estimate": float(np.mean(estimates)),
        "ci_low": float(np.nanquantile(values, 0.025)),
        "ci_high": float(np.nanquantile(values, 0.975)),
        "clusters": len(lookup),
        "target_split_evaluations": len(estimates),
    }


def membership(h, spec, labels, auxiliary, salt=None):
    P = partition_table(h, spec)
    inverse = original_runs(h)
    P["origin"] = P["run"].map(inverse)
    rows = []
    for dataset, group in labels.groupby("dataset", sort=True):
        group = group.sort_values("task").reset_index(drop=True)
        gids = group.gid.to_numpy(dtype=int)
        parts = [P[P["origin"] == r] for r in gids]
        targets = sorted(auxiliary[dataset])
        for split in SEEDS:
            train, test = known_indices(len(group), split)
            rng = np.random.default_rng(2000 + split)
            for target in targets:
                y = group.sites.map(lambda s, target=target: target in s.split(";")).to_numpy(dtype=int)
                if len(np.unique(y[test])) < 2:
                    continue
                aux = auxiliary[dataset][target]
                n_aux = aux["records"] * (1 + rng.uniform(-0.1, 0.1))
                w_aux = aux["width"] * (1 + rng.uniform(-0.1, 0.1))
                d_aux = aux["load_seconds"] * (1 + rng.uniform(-0.1, 0.1))
                visible = {target}
                if salt is not None:
                    visible.add(salted_pseudonym(salt, "site", target))
                # Token associations use ONLY labelled training tasks.
                token_pos, token_neg = {}, {}
                for i in train:
                    p = parts[i]
                    tokens = {(float(a), float(b)) for a, b in zip(p.n, p.b) if np.isfinite(a) and np.isfinite(b)}
                    counter = token_pos if y[i] else token_neg
                    for tok in tokens:
                        counter[tok] = counter.get(tok, 0) + 1
                features = []
                direct = []
                pos_n = max(1, int(y[train].sum()))
                neg_n = max(1, len(train) - int(y[train].sum()))
                for p in parts:
                    name = bool(set(p.site.astype(str)) & visible)
                    direct.append(name)

                    def distance(values, expected):
                        a = np.asarray(values, dtype=float)
                        a = a[np.isfinite(a)]
                        return float(np.min(np.abs(np.log1p(np.clip(a, 0, None)) - np.log1p(expected)))) if len(a) else np.nan

                    tok_scores = []
                    for a, b in zip(p.n, p.b):
                        if not np.isfinite(a) or not np.isfinite(b):
                            continue
                        t = (float(a), float(b))
                        tok_scores.append(
                            np.log((token_pos.get(t, 0) + 1) / (pos_n + 2)) - np.log((token_neg.get(t, 0) + 1) / (neg_n + 2))
                        )
                    features.append(
                        [
                            float(name),
                            distance(p.n, n_aux),
                            distance(p.w, w_aux),
                            distance(p.dur, d_aux),
                            len(p),
                            np.nansum(p.n),
                            np.nansum(p.b),
                            max(tok_scores, default=0.0),
                        ]
                    )
                X = matrix(features)
                # Clear/public-salt names permit direct lookup without learning.
                all_names = set().union(*(set(p.site.astype(str)) for p in parts))
                public_roster = set(targets)
                if salt is not None:
                    public_roster |= {salted_pseudonym(salt, "site", s) for s in targets}
                direct_available = bool(all_names) and all_names <= public_roster
                if direct_available:
                    pred = np.asarray(direct, dtype=float)[test]
                elif len(np.unique(y[train])) < 2:
                    pred = np.full(len(test), y[train].mean())
                else:
                    clf = model(split)
                    clf.fit(X[train], y[train])
                    pred = clf.predict_proba(X[test])[:, 1]
                for i, score in zip(test, pred):
                    rows.append(
                        {
                            "dataset": dataset,
                            "task": str(group.task.iloc[i]),
                            "target": target,
                            "split": split,
                            "y": int(y[i]),
                            "score": float(score),
                        }
                    )
    predictions = pd.DataFrame(rows)
    return auc_interval(predictions), predictions


def analyst_profiles(h, labels, salt=None):
    inverse = original_runs(h)
    a = h.act.copy()
    a["origin"] = a.run.map(inverse)
    if "b_start" in a:
        a["s"] = a.b_start.astype(float) * h.meta.get("delta", 1.0)
        a["e"] = a.b_end.astype(float) * h.meta.get("delta", 1.0)
    elif "start" in a and pd.api.types.is_numeric_dtype(a.start):
        a["s"] = a.start
        a["e"] = a.end
    else:
        a["s"] = np.nan
        a["e"] = np.nan
    errors = h.ent[h.ent.type == "tool_error"].groupby("run").size()
    errors = {inverse[int(r)]: int(n) for r, n in errors.items()}
    names = list(ANALYST_PROFILES)
    data = labels.sort_values("task").reset_index(drop=True)
    columns = []
    for gid in data.gid:
        sub = a[a.origin == gid]
        llm = sub[sub.type.isin(["agent_step", "agent_report"])]
        columns.append(
            [
                len(sub),
                len(llm),
                float((sub.e - sub.s).sum()),
                float((llm.e - llm.s).median()),
                float(sub.e.max() - sub.s.min()),
                errors.get(int(gid), 0),
                int((sub.type == "profile_feature").sum()),
                int((sub.type == "train_model").sum()),
            ]
        )
    X = matrix(columns)
    y = np.array([names.index(s) for s in data.user])
    users = {inverse[int(r.run)]: str(r.user) for r in h.runs.itertuples()}
    direct = {name: i for i, name in enumerate(names)}
    if salt is not None:
        direct.update({salted_pseudonym(salt, "user", n): i for i, n in enumerate(names)})
    predictions = []
    for split in SEEDS:
        train, test = known_indices(len(data), split)
        if len(np.unique(y[train])) < 2:
            pred = np.full(len(test), y[train][0])
        else:
            clf = model(split)
            clf.fit(X[train], y[train])
            pred = clf.predict(X[test])
        majority = int(np.bincount(y[train], minlength=len(names)).argmax())
        for i, p in zip(test, pred):
            if users.get(int(data.gid.iloc[i])) in direct:
                p = direct[users[int(data.gid.iloc[i])]]
            predictions.append(
                {
                    "dataset": data.dataset.iloc[i],
                    "task": str(data.task.iloc[i]),
                    "split": split,
                    "correct": float(int(p) == y[i]),
                    "baseline": float(majority == y[i]),
                }
            )
    frame = pd.DataFrame(predictions)
    result = accuracy_interval(frame)
    result["baseline"] = accuracy_interval(frame, "baseline")["estimate"]
    result["interpretation"] = "simulated analyst-profile attribution, not real-person re-identification"
    return result, frame


def partition_features(h, spec):
    p = partition_table(h, spec)
    return p, matrix(p[["n", "b", "w", "dur", "pos"]])


def size_classes(h, shadow, raw, raw_shadow, spec, key, shadow_key, thresholds, dataset_by_part):
    p, X = partition_features(h, spec)
    ps, Xs = partition_features(shadow, spec)
    target_map = node_map(raw, h, key)
    shadow_map = node_map(raw_shadow, shadow, shadow_key)
    source = raw.ent[raw.ent.type == spec.source_type]
    sources = raw_shadow.ent[raw_shadow.ent.type == spec.source_type]
    truth = {
        target_map[int(r.id)]: (str(r.part_id), int(np.searchsorted(thresholds, float(r.n_records), side="right")))
        for r in source.itertuples()
    }
    shadow_truth = {
        shadow_map[int(r.id)]: int(np.searchsorted(thresholds, float(r.n_records), side="right")) for r in sources.itertuples()
    }
    valid = np.array([int(i) in truth for i in p.id])
    vs = np.array([int(i) in shadow_truth for i in ps.id])
    p = p[valid].reset_index(drop=True)
    X = X[valid]
    Xs = Xs[vs]
    if not len(p) or not len(Xs):
        return {"estimate": None, "ci_low": None, "ci_high": None, "clusters": 0}, pd.DataFrame()
    ys = np.array([shadow_truth[int(i)] for i in ps.loc[vs, "id"]])
    yt = np.array([truth[int(i)][1] for i in p.id])
    majority = int(np.bincount(ys, minlength=3).argmax())
    rows = []
    for split in SEEDS:
        clf = model(split)
        clf.fit(Xs, ys)
        pred = clf.predict(X)
        for i, (correct, base) in enumerate(zip(pred == yt, majority == yt)):
            part = truth[int(p.id.iloc[i])][0]
            rows.append(
                {
                    "dataset": dataset_by_part[part],
                    "source_version": part,
                    "split": split,
                    "correct": float(correct),
                    "baseline": float(base),
                }
            )
    frame = pd.DataFrame(rows)
    result = accuracy_interval(frame, cluster="source_version")
    result["baseline"] = accuracy_interval(frame, "baseline", cluster="source_version")["estimate"]
    result["class_thresholds_records"] = list(map(float, thresholds))
    result["target_classes_observed"] = sorted(set(map(int, yt)))
    return result, frame


def dictionary_candidates(datasets):
    return [
        canonical(initial_messages(ds, list(sites), 42, wf, True, profile))
        for ds in datasets
        for sites in itertools.combinations(ds.site_ids, 10)
        for wf in ("explore", "regression")
        for profile in ANALYST_PROFILES
    ]


def dictionary(h, raw, key, candidates, salt=None):
    published = set(h.ent.loc[h.ent.type == "prompt", "text"].astype(str))
    tokens = set(candidates) if salt is None else {salted_pseudonym(salt, "text", s) for s in candidates}
    ids = node_map(raw, h, key)
    initial = raw.ent[raw.ent.type == "prompt"].sort_values("id").groupby("run", sort=False).first()["id"]
    ids_initial = {ids[int(i)] for i in initial}
    visible_initial = set(h.ent.loc[h.ent.id.isin(ids_initial), "text"].astype(str))
    return {
        "candidate_count": len(tokens),
        "all_distinct_prompts": len(published),
        "all_prompt_recovery": len(published & tokens) / max(1, len(published)),
        "initial_distinct_prompts": len(visible_initial),
        "initial_prompt_recovery": len(visible_initial & tokens) / max(1, len(visible_initial)),
        "dictionary_source": "public initial templates; no held-out captured text enters the candidate set",
    }


def load_order(h, raw, spec, key, labels):
    if "rank" in h.act:
        positions = h.act["rank"]
    elif "start" in h.act and pd.api.types.is_numeric_dtype(h.act.start):
        positions = h.act.start
    else:
        return {"estimate": None, "ci_low": None, "ci_high": None, "clusters": 0}, pd.DataFrame()
    sources = set(raw.ent.loc[raw.ent.type == spec.source_type, "id"])
    successful = set(raw.edge.loc[(raw.edge.kind == "used") & raw.edge.dst.isin(sources), "src"])
    original = raw.act[(raw.act.type == "load_partition") & raw.act.id.isin(successful)]
    ids = node_map(raw, h, key)
    true_sites = {ids[int(r.id)]: r.site for r in original.itertuples()}
    inverse = original_runs(h)
    selected = h.act[h.act.id.isin(true_sites)].copy()
    selected["position"] = positions.loc[selected.index]
    selected["origin"] = selected.run.map(inverse)
    lookup = labels.set_index("gid")
    rows = []
    for origin, part in selected.groupby("origin"):
        if origin not in lookup.index:
            continue
        part = part.sort_values("position", kind="stable")
        sites = [true_sites[int(i)] for i in part.id]
        if not len(sites):
            continue
        rows.append(
            {
                "dataset": lookup.loc[origin, "dataset"],
                "task": str(lookup.loc[origin, "task"]),
                "correct": float(np.mean(np.array(sites) == np.array(sorted(sites)))),
                "chance": 1 / len(sites),
            }
        )
    frame = pd.DataFrame(rows)
    result = accuracy_interval(frame)
    result["chance"] = float(frame.chance.mean()) if len(frame) else None
    result["assumption"] = "adversary knows the contributing roster and guesses alphabetical load order"
    return result, frame
