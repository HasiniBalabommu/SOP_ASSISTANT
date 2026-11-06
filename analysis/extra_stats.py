"""Stored-data analyses added for MINE3 (audit rows 20, 64, 65, 69, 83, 88).
Run from the sop_assistant folder:  python <this file> <out.json>"""
import json, re, statistics as st, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from eval.refusal import is_refusal
from eval.analyze import is_echo

RES = Path("eval/results_v2")
B = 10000
rng = np.random.default_rng(2026)


def load(name):
    p = RES / f"{name}.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    for r in rows:
        r["refusal_text"] = is_refusal(r["answer"])
    return rows


non_answer = lambda r: bool(r["declined"] or r["refusal_text"])
fabricated = lambda r: not non_answer(r) and not is_echo(r)
declined = lambda r: bool(r["declined"])


def per_q(rows, cfg, pred, scope):
    """{question: fraction of its repeats with the outcome}"""
    d = defaultdict(list)
    for r in rows:
        if r["config"] == cfg and r["in_scope"] == scope:
            d[r["id"]].append(bool(pred(r)))
    return {q: sum(v) / len(v) for q, v in d.items()}


def boot_ci(values, clusters=None):
    """Percentile bootstrap of a mean, resampling questions (or the given clusters)."""
    v = np.asarray(values, float)
    if clusters is None:
        idx = rng.integers(0, len(v), (B, len(v)))
        m = v[idx].mean(1)
    else:
        labels = sorted(set(clusters))
        groups = [v[[i for i, c in enumerate(clusters) if c == l]] for l in labels]
        sums = np.array([g.sum() for g in groups]); ns = np.array([len(g) for g in groups])
        idx = rng.integers(0, len(labels), (B, len(labels)))
        m = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return [100 * float(lo), 100 * float(hi)]


def rate(tag, rows, cfg, pred, scope):
    q = per_q(rows, cfg, pred, scope)
    v = list(q.values())
    return {"pct": 100 * float(np.mean(v)), "n_q": len(v), "ci": boot_ci(v)} if v else None


def diff(a_rows, a_cfg, b_rows, b_cfg, pred, scope, clusters=None):
    """Risk difference B - A in percentage points, paired by question."""
    a, b = per_q(a_rows, a_cfg, pred, scope), per_q(b_rows, b_cfg, pred, scope)
    ids = sorted(set(a) & set(b))
    d = [b[q] - a[q] for q in ids]
    out = {"rd": 100 * float(np.mean(d)), "n_q": len(ids), "ci": boot_ci(d), "a": 100 * float(np.mean([a[q] for q in ids])), "b": 100 * float(np.mean([b[q] for q in ids]))}
    if clusters:
        out["ci_article"] = boot_ci(d, [clusters[q] for q in ids])
    return out


def signflip(d, n=200000):
    """Two-sided sign-flip permutation p of the mean of d (exact when few non-zero values)."""
    d = np.asarray([x for x in d if x != 0], float)
    if not len(d):
        return 1.0
    obs = abs(d.sum())
    if len(d) <= 20:
        signs = ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1) * 2 - 1
        return float(np.mean(np.abs((signs * d).sum(1)) >= obs - 1e-12))
    signs = rng.choice([-1, 1], (n, len(d)))
    return float((np.sum(np.abs((signs * d).sum(1)) >= obs - 1e-12) + 1) / (n + 1))


g3, g1 = load("g3b_core"), load("g1b_core")
x13, x31 = load("x1b_verify3b"), load("x3b_verify1b")
mc, mcs = load("g3b_minicheck"), load("squad_g3b_minicheck")
s3, s1 = load("squad_g3b"), load("squad_g1b")
nov = load("g3b_novel")
out = {}

# Row 64: intervals that resample questions
out["cluster_ci"] = {
    "1b_no_verify_fab": rate("", g1, "no_verify", fabricated, False),
    "1b_full_fab": rate("", g1, "full", fabricated, False),
    "3b_no_verify_fab": rate("", g3, "no_verify", fabricated, False),
    "3b_full_fab": rate("", g3, "full", fabricated, False),
    "x13_fab": rate("", x13, "full", fabricated, False),
    "x31_fab": rate("", x31, "full", fabricated, False),
    "mc_fab": rate("", mc, "checker_verify", fabricated, False),
    "3b_full_fd": rate("", g3, "full", declined, True),
    "1b_full_fd": rate("", g1, "full", declined, True),
    "x13_fd": rate("", x13, "full", declined, True),
    "mc_fd": rate("", mc, "checker_verify", declined, True),
}

# Row 65: risk differences with question-bootstrap intervals
out["risk_diff"] = {
    "1B fabrication, verifier vs no check": diff(g1, "no_verify", g1, "full", fabricated, False),
    "3B fabrication, verifier vs no check": diff(g3, "no_verify", g3, "full", fabricated, False),
    "3B false declines, verifier vs no check": diff(g3, "no_verify", g3, "full", declined, True),
    "1B generator fabrication, 3B verifier vs no check": diff(g1, "no_verify", x13, "full", fabricated, False),
    "1B generator false declines, 3B verifier vs no check": diff(g1, "no_verify", x13, "full", declined, True),
    "3B generator fabrication, 1B verifier vs no check": diff(g3, "no_verify", x31, "full", fabricated, False),
    "3B fabrication, MiniCheck vs no check": diff(g3, "no_verify", mc, "checker_verify", fabricated, False),
}

# Row 83: SQuAD questions grouped by article
Q = json.loads(Path("eval/squad/questions_squad.json").read_text(encoding="utf-8"))
art = {q["id"]: q.get("article") or (q["expected_sources"][0] if q.get("expected_sources") else q.get("title")) for q in Q}
if any(v is None for v in art.values()):
    raise SystemExit("no article for some SQuAD questions: " + str(list(Q[0].keys())))
out["squad_articles"] = Counter(art.values()).most_common()
out["risk_diff"]["SQuAD 3B fabrication, verifier vs no check"] = diff(s3, "no_verify", s3, "full", fabricated, False, art)
out["risk_diff"]["SQuAD 3B unanswered, verifier vs no check"] = diff(s3, "no_verify", s3, "full", non_answer, True, art)
out["risk_diff"]["SQuAD 1B fabrication, verifier vs no check"] = diff(s1, "no_verify", s1, "full", fabricated, False, art)

a, b = per_q(s3, "no_verify", fabricated, False), per_q(s3, "full", fabricated, False)
by_art = defaultdict(list)
for q in a:
    by_art[art[q]].append(a[q] - b[q])
art_means = {k: float(np.mean(v)) for k, v in by_art.items()}
out["squad_cluster"] = {"n_articles_with_unanswerable": len(art_means), "per_article_reduction": art_means,
                        "reduced": sum(v > 0 for v in art_means.values()), "increased": sum(v < 0 for v in art_means.values()),
                        "tied": sum(v == 0 for v in art_means.values()), "p_article_permutation": signflip(list(art_means.values()))}

# Row 69: size x verification interaction on SQuAD (one run, so per-question differences are -1, 0 or 1)
a3, b3 = per_q(s3, "no_verify", fabricated, False), per_q(s3, "full", fabricated, False)
a1, b1 = per_q(s1, "no_verify", fabricated, False), per_q(s1, "full", fabricated, False)
ids = sorted(set(a3) & set(a1))
dd = [(a3[q] - b3[q]) - (a1[q] - b1[q]) for q in ids]
out["squad_interaction"] = {"n_q": len(ids), "nonzero": sum(x != 0 for x in dd), "mean_red_3b": float(np.mean([a3[q] - b3[q] for q in ids])),
                            "mean_red_1b": float(np.mean([a1[q] - b1[q] for q in ids])), "p": signflip(dd)}
ad = [dd[i] for i in range(len(ids))]
by_a = defaultdict(list)
for q, x in zip(ids, ad):
    by_a[art[q]].append(x)
out["squad_interaction"]["p_article"] = signflip([float(np.mean(v)) for v in by_a.values()])

# Row 20: coverage of the claim-level checks
def claims_cover(rows, cfg):
    sel = [r for r in rows if r["config"] == cfg]
    n = len(sel)
    lens = [len(r.get("claims") or []) for r in sel]
    capped = sum(l >= 8 for l in lens)
    none = sum(l == 0 for l in lens)
    long_drafts = 0; skipped = 0; visible = 0
    for r in sel:
        if r["declined"]:
            continue
        visible += 1
        text = r["answer"].split("\n\n", 1)[1] if r["answer"].startswith("Note: parts of this answer") else r["answer"]
        parts = [p.strip(" -*\t") for p in re.split(r"(?<=[.!?])\s+|\n+", text)]
        real = [p for p in parts if len(p.split()) >= 4]
        long_drafts += len(real) > 8
        skipped += any(0 < len(p.split()) < 4 for p in parts)
    return {"n": n, "at_cap": capped, "no_sentence": none, "median_claims": st.median(lens), "shown": visible,
            "shown_over_cap": long_drafts, "shown_with_short_sentence": skipped}


out["claim_cover"] = {"claim_verify": claims_cover(nov, "claim_verify"), "minicheck": claims_cover(mc, "checker_verify"),
                      "minicheck_squad": claims_cover(mcs, "checker_verify")}

# Row 88: run order and conditions
order = []
for name in ["g1b_core", "g3b_core", "g3b_novel", "g3b_poison", "squad_g3b", "x1b_verify3b", "x3b_verify1b", "squad_g3b_base",
             "squad_g1b", "g1b_legacy_shared", "g1b_q4_core", "g3b_minicheck", "squad_g3b_minicheck"]:
    rows = load(name)
    if not rows:
        continue
    ts = sorted(datetime.fromisoformat(r["time_utc"]) for r in rows)
    gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    lat = [r["latency_sec"] for r in rows]
    order.append({"run": name, "rows": len(rows), "start": ts[0].isoformat(timespec="minutes"), "end": ts[-1].isoformat(timespec="minutes"),
                  "gaps_over_10min": sum(g > 600 for g in gaps), "longest_gap_min": round(max(gaps) / 60, 1) if gaps else 0,
                  "median_s": round(st.median(lat), 1), "lat_over_300s": sum(l > 300 for l in lat)})
out["run_order"] = order
# drift: the same configuration, first against last repeat
def drift(rows, cfg):
    m = {}
    for x in sorted({r["repeat"] for r in rows}):
        m[x] = st.median(r["latency_sec"] for r in rows if r["config"] == cfg and r["repeat"] == x)
    return m


out["latency_drift"] = {"3b_full": drift(g3, "full"), "1b_full": drift(g1, "full"), "3b_no_verify": drift(g3, "no_verify")}

json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ("cluster_ci", "risk_diff", "squad_cluster", "squad_interaction", "claim_cover", "latency_drift")}, indent=0, default=str)[:7000])
print(*[str(o) for o in order], sep="\n")
