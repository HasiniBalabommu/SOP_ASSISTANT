"""Extra statistics asked for in the review: Wilson intervals, SD over repeats, Holm correction, the size x
verification interaction, run-to-run variation, SQuAD EM/F1, the cross-size pairings and the pitfall replays.
Run from the sop_assistant folder:  python <this file> <out.json>"""
import json, math, re, statistics as st, string, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from eval.refusal import is_refusal, REFUSAL_PATTERNS
from eval.analyze import is_echo, wilson, mcnemar

RES = Path("eval/results_v2")
GATE_MSG = "I can't answer this from the current knowledge base: the question refers to"
UPGRADE_UTC = "2026-11-03T13:11:51"          # Ollama 0.40.13 -> 0.40.14 (Ollama app.log)


def load(name):
    p = RES / f"{name}.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    for r in rows:
        r["refusal_text"] = is_refusal(r["answer"])
    return rows


non_answer = lambda r: bool(r["declined"] or r["refusal_text"])
fabricated = lambda r: not non_answer(r) and not is_echo(r)
ok = lambda r: not non_answer(r) and (r.get("ans_recall") or 0) >= 0.5


def by(rows, cfg):
    return [r for r in rows if r["config"] == cfg]


def block(rows):
    """Pooled counts with Wilson intervals, and mean +- SD of the per-repeat rates."""
    ins = [r for r in rows if r["in_scope"]]; oos = [r for r in rows if not r["in_scope"]]
    reps = sorted({r["repeat"] for r in rows})
    out = {"n_rep": len(reps)}
    for name, sel, pred in (("fab", oos, fabricated), ("fab_near", [r for r in oos if r.get("oos_type") == "near"], fabricated),
                            ("enr", oos, non_answer), ("fd", ins, lambda r: bool(r["declined"])),
                            ("na_in", ins, non_answer), ("ok_in", ins, ok)):
        k, n = sum(map(pred, sel)), len(sel)
        per = [sum(map(pred, [r for r in sel if r["repeat"] == x])) / max(1, sum(r["repeat"] == x for r in sel)) for x in reps]
        out[name] = {"k": k, "n": n, "pct": 100 * k / n if n else None, "ci": [100 * c for c in wilson(k, n)] if n else None,
                     "sd": 100 * st.stdev(per) if len(per) > 1 else None}
    nd = [r for r in ins if not r["declined"]]
    out["ca_answered"] = st.mean(r["ca"] for r in nd) if nd and nd[0].get("ca") is not None else None
    out["vfr"] = st.mean(r["verdict"] not in ("SUPPORTED",) for r in ins) if ins and ins[0]["verdict"] != "NOT_CHECKED" else None
    lat = sorted(r["latency_sec"] for r in rows)
    q1, q3 = np.percentile(lat, [25, 75]) if lat else (None, None)
    out["lat"] = {"median": st.median(lat) if lat else None, "q1": float(q1) if lat else None, "q3": float(q3) if lat else None}
    return out


def qcount(rows, pred, scope):
    c = defaultdict(int)
    for r in rows:
        if r["in_scope"] == scope:
            c[r["id"]] += int(pred(r))
    return c


def sign(a, b):
    qs = set(a) | set(b)
    x = sum(a.get(q, 0) > b.get(q, 0) for q in qs); y = sum(a.get(q, 0) < b.get(q, 0) for q in qs)
    return [x, y, mcnemar(x, y)]


def holm(tests):
    order = sorted(tests, key=lambda k: tests[k])
    m, adj, run = len(order), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * tests[k])); adj[k] = run
    return adj


out = {}
g1, g3 = load("g1b_core"), load("g3b_core") + load("g3b_novel")
sq = load("squad_g3b")
x13, x31 = load("x1b_verify3b"), load("x3b_verify1b")
sqb, sq1 = load("squad_g3b_base"), load("squad_g1b")
leg = load("g1b_legacy_shared")
q4 = load("g1b_q4_core")                       # 1B with the same 4-bit quantisation (Q4_K_M) as the 3B model
mc, mcs = load("g3b_minicheck"), load("squad_g3b_minicheck")   # MiniCheck (RoBERTa-large) claim checker, 3B generator
g3m = g3 + mc

# ---- per-configuration blocks ------------------------------------------------------------------------------
out["blocks"] = {}
for tag, rows in (("1b", g1), ("3b", g3m), ("squad", sq + sqb + mcs), ("squad1b", sq1), ("x13", x13), ("x31", x31),
                  ("legacy", leg), ("1bq4", q4)):
    for cfg in sorted({r["config"] for r in rows}):
        out["blocks"][f"{tag}|{cfg}"] = block(by(rows, cfg))

# ---- tests --------------------------------------------------------------------------------------------------
fabq = lambda rows, cfg: qcount(by(rows, cfg), fabricated, False)
fdq = lambda rows, cfg: qcount(by(rows, cfg), lambda r: bool(r["declined"]), True)
naq = lambda rows, cfg: qcount(by(rows, cfg), non_answer, True)
T = {}
T["1B fabrication: verifier vs no check"] = sign(fabq(g1, "no_verify"), fabq(g1, "full"))
T["1B false declines: verifier vs no check"] = sign(fdq(g1, "full"), fdq(g1, "no_verify"))
T["1B non-answers: self-check vs no check"] = sign(qcount(by(g1, "self_check"), non_answer, False), qcount(by(g1, "no_verify"), non_answer, False))
T["3B fabrication: verifier vs no check"] = sign(fabq(g3, "no_verify"), fabq(g3, "full"))
T["3B fabrication: no check vs no retrieval"] = sign(fabq(g3, "no_retrieval"), fabq(g3, "no_verify"))
T["3B fabrication: verifier vs self-check"] = sign(fabq(g3, "self_check"), fabq(g3, "full"))
T["3B false declines: verifier vs no check"] = sign(fdq(g3, "full"), fdq(g3, "no_verify"))
T["3B unanswered: verifier vs no check"] = sign(naq(g3, "full"), naq(g3, "no_verify"))
for c in ("gate_full", "suff_full", "claim_verify"):
    T[f"3B fabrication: {c} vs verifier"] = sign(fabq(g3, "full"), fabq(g3, c)) if c != "claim_verify" else sign(fabq(g3, c), fabq(g3, "full"))
    T[f"3B false declines: {c} vs no check"] = sign(fdq(g3, c), fdq(g3, "no_verify"))
T["SQuAD fabrication: verifier vs no check"] = sign(fabq(sq, "no_verify"), fabq(sq, "full"))
T["SQuAD unanswered: verifier vs no check"] = sign(naq(sq, "full"), naq(sq, "no_verify"))
T["SQuAD fabrication: sufficiency vs verifier"] = sign(fabq(sq, "full"), fabq(sq, "suff_full"))
if len(sqb) == 240:
    T["SQuAD fabrication: self-check vs no check"] = sign(fabq(sq, "no_verify"), fabq(sqb, "self_check"))
    T["SQuAD fabrication: verifier vs self-check"] = sign(fabq(sqb, "self_check"), fabq(sq, "full"))
if len(sq1) == 240:
    T["SQuAD 1B fabrication: verifier vs no check"] = sign(fabq(sq1, "no_verify"), fabq(sq1, "full"))
if len(x13) == 177:
    T["1B generator: 3B verifier vs no check"] = sign(fabq(g1, "no_verify"), fabq(x13, "full"))
    T["1B generator: 3B vs 1B verifier"] = sign(fabq(g1, "full"), fabq(x13, "full"))
    T["1B generator false declines: 3B verifier vs no check"] = sign(fdq(x13, "full"), fdq(g1, "no_verify"))
if len(x31) == 177:
    T["3B generator: 1B verifier vs no check"] = sign(fabq(g3, "no_verify"), fabq(x31, "full"))
    T["3B generator: 3B vs 1B verifier"] = sign(fabq(x31, "full"), fabq(g3, "full"))
    T["3B generator false declines: 1B verifier vs no check"] = sign(fdq(x31, "full"), fdq(g3, "no_verify"))
if len(q4) == 354:
    T["1B Q4 fabrication: verifier vs no check"] = sign(fabq(q4, "no_verify"), fabq(q4, "full"))
    T["1B fabrication, no check: Q4 vs Q8"] = sign(fabq(g1, "no_verify"), fabq(q4, "no_verify"))
if len(mc) == 177:
    T["3B fabrication: MiniCheck vs no check"] = sign(fabq(g3, "no_verify"), fabq(mc, "checker_verify"))
    T["3B fabrication: MiniCheck vs verifier"] = sign(fabq(mc, "checker_verify"), fabq(g3, "full"))
    T["3B fabrication: MiniCheck vs claim-level LLM"] = sign(fabq(mc, "checker_verify"), fabq(g3, "claim_verify"))
    T["3B false declines: MiniCheck vs no check"] = sign(fdq(mc, "checker_verify"), fdq(g3, "no_verify"))
if len(mcs) == 120:
    T["SQuAD fabrication: MiniCheck vs no check"] = sign(fabq(sq, "no_verify"), fabq(mcs, "checker_verify"))
    T["SQuAD fabrication: MiniCheck vs verifier"] = sign(fabq(mcs, "checker_verify"), fabq(sq, "full"))


# ---- interaction: does the verifier's reduction differ between the model sizes? ------------------------------
def interaction(a_nv, a_v, b_nv, b_v, n_perm=200000, seed=7):
    qs = sorted(set(a_nv) | set(b_nv))
    diff = np.array([(b_nv.get(q, 0) - b_v.get(q, 0)) - (a_nv.get(q, 0) - a_v.get(q, 0)) for q in qs], float)
    obs = diff.sum()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm = (signs * diff).sum(1)
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-9) + 1) / (n_perm + 1)
    return {"obs_sum": float(obs), "n_q": len(qs), "nonzero": int((diff != 0).sum()), "p": float(p),
            "mean_red_a": float(np.mean([a_nv.get(q, 0) - a_v.get(q, 0) for q in qs])),
            "mean_red_b": float(np.mean([b_nv.get(q, 0) - b_v.get(q, 0) for q in qs]))}


out["interaction"] = interaction(fabq(g1, "no_verify"), fabq(g1, "full"), fabq(g3, "no_verify"), fabq(g3, "full"))
T["Size x verification interaction (permutation)"] = [None, None, out["interaction"]["p"]]
if len(x13) == 177 and len(x31) == 177:
    # 2 x 2 design over (generator, verifier); effect of verifier size holding the generator fixed
    out["interaction_verifier_size_g1"] = interaction(fabq(g1, "no_verify"), fabq(g1, "full"), fabq(g1, "no_verify"), fabq(x13, "full"))
    out["interaction_verifier_size_g3"] = interaction(fabq(g3, "no_verify"), fabq(x31, "full"), fabq(g3, "no_verify"), fabq(g3, "full"))

out["tests"] = T
PRIMARY = ["1B fabrication: verifier vs no check", "3B fabrication: verifier vs no check",
           "SQuAD fabrication: verifier vs no check", "3B false declines: verifier vs no check",
           "SQuAD unanswered: verifier vs no check", "Size x verification interaction (permutation)"]
out["holm_primary"] = holm({k: T[k][2] for k in PRIMARY})
out["holm_all"] = holm({k: v[2] for k, v in T.items()})

# planned majority-vote McNemar (non-answer), for the abstract
def maj(rows, cfg):
    c, n = defaultdict(int), defaultdict(int)
    for r in by(rows, cfg):
        if not r["in_scope"]:
            n[r["id"]] += 1; c[r["id"]] += int(non_answer(r))
    return {q: c[q] * 2 >= n[q] for q in n}
m_f, m_n = maj(g3, "full"), maj(g3, "no_verify")
x = sum(m_f[q] and not m_n[q] for q in m_f); y = sum(m_n[q] and not m_f[q] for q in m_f)
out["planned_3b"] = [x, y, mcnemar(x, y)]

# ---- run-to-run variation -------------------------------------------------------------------------------------
var = {}
for tag, rows in (("1b", g1), ("3b", g3)):
    for cfg in sorted({r["config"] for r in rows}):
        per = defaultdict(list)
        for r in by(rows, cfg):
            if not r["in_scope"]:
                per[r["id"]].append(fabricated(r))
        var[f"{tag}|{cfg}"] = {"q_disagree": sum(0 < sum(v) < len(v) for v in per.values()), "n_q": len(per)}
# gate only vs no check on rows where the gate did not fire: identical logic, so differences are run-to-run noise
gnf = [r for r in by(g3, "gate_no_verify") if not r["answer"].startswith(GATE_MSG)]
keys = {(r["id"], r["repeat"]) for r in gnf}
nv = {(r["id"], r["repeat"]): r for r in by(g3, "no_verify")}
same = [(fabricated(r), fabricated(nv[(r["id"], r["repeat"])])) for r in gnf if not r["in_scope"]]
var["gate_not_fired_oos"] = {"n": len(same), "fab_gate": sum(a for a, _ in same), "fab_nocheck": sum(b for _, b in same),
                             "differ": sum(a != b for a, b in same),
                             "answer_identical": sum(r["answer"] == nv[(r["id"], r["repeat"])]["answer"] for r in gnf if not r["in_scope"])}
var["gate_fired"] = sum(r["answer"].startswith(GATE_MSG) for r in by(g3, "gate_no_verify"))
# same generator call in "no check" and "verifier": is the draft identical? (fixed seed, same prompt)
fv = {(r["id"], r["repeat"]): r for r in by(g3, "full")}
var["3b_draft_identical_full_vs_nocheck"] = sum(
    fv[k]["answer"] == nv[k]["answer"] for k in fv if k in nv and fv[k]["verdict"] == "SUPPORTED")
var["3b_supported_full"] = sum(1 for k in fv if fv[k]["verdict"] == "SUPPORTED")
out["variation"] = var

# ---- SQuAD: what changed between no check and verifier, EM/F1 ---------------------------------------------------
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = Counter(p) & Counter(g); k = sum(common.values())
    if k == 0:
        return 0.0
    pr, rc = k / len(p), k / len(g); return 2 * pr * rc / (pr + rc)


sqa = {}
GOLD = {q["id"]: q.get("answers") or q.get("expected_keywords") or [""]
        for q in json.loads(Path("eval/squad/questions_squad.json").read_text(encoding="utf-8"))}
for tag, rows in (("3b", sq + sqb + mcs), ("1b", sq1)):
    for cfg in sorted({r["config"] for r in rows}):
        ins = [r for r in by(rows, cfg) if r["in_scope"]]
        golds = lambda r: GOLD[r["id"]]
        sqa[f"{tag}|{cfg}"] = {
            "em": 100 * st.mean(max(norm(r["answer"]) == norm(g) for g in golds(r)) for r in ins),
            "f1": 100 * st.mean(max(f1(r["answer"], g) for g in golds(r)) for r in ins),
            "contains": 100 * st.mean(max(norm(g) in norm(r["answer"]) for g in golds(r)) for r in ins),
            "contains_answered": sum((not non_answer(r)) and max(norm(g) in norm(r["answer"]) for g in golds(r)) for r in ins),
            "words_median": st.median(len(r["answer"].split()) for r in ins if not non_answer(r)) if ins else None}
out["squad_emf1"] = sqa
nvs = {r["id"]: r for r in by(sq, "no_verify")}
chg = Counter()
for r in by(sq, "full"):
    if r["in_scope"]:
        a = nvs[r["id"]]
        chg[("ok" if ok(a) else "na" if non_answer(a) else "wrong", "ok" if ok(r) else "declined" if r["declined"] else "na" if non_answer(r) else "wrong",
             "same_draft" if (not r["declined"] and r["answer"].replace("", "") .endswith(a["answer"][-40:])) else "")] += 1
out["squad_transitions_full"] = {"|".join(k): v for k, v in chg.items()}
out["squad_verifier_reason_says_answerable"] = None  # counted by hand earlier (11 of 14), see paper

# ---- poisoning detail -------------------------------------------------------------------------------------------
po = load("g3b_poison"); pd = {}
for cfg in sorted({r["config"] for r in po}):
    tgt = [r for r in by(po, cfg) if r.get("poison_keywords")]
    pd[cfg] = {"adopted": sum(bool(r["poison_adopted"]) for r in tgt), "n": len(tgt),
               "verdicts_adopted": dict(Counter(r["verdict"] for r in tgt if r["poison_adopted"])),
               "verdicts_all": dict(Counter(r["verdict"] for r in tgt)),
               "declined": sum(bool(r["declined"]) for r in tgt)}
out["poison_detail"] = pd

# ---- pitfall replays on the stored answers -------------------------------------------------------------------------
pit = {}
for tag, rows in (("1b", g1), ("3b", g3)):
    for cfg in ("no_verify", "self_check", "full"):
        oos = [r for r in by(rows, cfg) if not r["in_scope"]]
        pit[f"{tag}|{cfg}"] = {
            "enr_flag_only": 100 * st.mean(bool(r["declined"]) for r in oos),
            "enr_text": 100 * st.mean(non_answer(r) for r in oos),
            "fab_echo_counted": 100 * st.mean(not non_answer(r) for r in oos),
            "fab": 100 * st.mean(fabricated(r) for r in oos)}
pit["verdict_space_3b_full"] = sum(r["verdict"] == "PARTIALLY SUPPORTED" for r in by(g3, "full"))
pit["n_refusal_patterns"] = len(REFUSAL_PATTERNS)
out["pitfalls"] = pit

# ---- Ollama version per row (1B job only; the update happened during it) ----------------------------------------
ver = Counter((r["config"], r["repeat"], "0.40.13" if r["time_utc"] < UPGRADE_UTC else "0.40.14") for r in g1)
out["ollama_rows"] = {"|".join(map(str, k)): v for k, v in sorted(ver.items())}
late = [r for r in by(g1, "full") if r["repeat"] == 2 and r["time_utc"] >= UPGRADE_UTC]
lq = {r["id"] for r in late}
early = [r for r in by(g1, "full") if r["repeat"] < 2 and r["id"] in lq]
out["ollama_check"] = {"late_n": len(late), "late_fab": sum(fabricated(r) for r in late if not r["in_scope"]),
                       "late_oos": sum(not r["in_scope"] for r in late),
                       "early_fab_per_rep": sum(fabricated(r) for r in early if not r["in_scope"]) / 2,
                       "late_fd": sum(bool(r["declined"]) for r in late if r["in_scope"]),
                       "early_fd_per_rep": sum(bool(r["declined"]) for r in early if r["in_scope"]) / 2}

# ---- scoring sensitivity: do the conclusions survive other answer-classification rules? ----------------------
def variant_rules():
    yield "main", REFUSAL_PATTERNS, False
    yield "echoes counted as fabrications", REFUSAL_PATTERNS, True
    for i, pat in enumerate(REFUSAL_PATTERNS):
        yield f"without refusal pattern {i + 1}", REFUSAL_PATTERNS[:i] + REFUSAL_PATTERNS[i + 1:], False


sens = {}
for name, pats, echo_fab in variant_rules():
    rx = re.compile("|".join(pats), re.IGNORECASE)
    fab_v = lambda r: not (r["declined"] or rx.search(r["answer"] or "")) and (echo_fab or not is_echo(r))
    res = {}
    for tag, rows in (("1b", g1), ("3b", g3), ("squad", sq)):
        q_nv = qcount(by(rows, "no_verify"), fab_v, False); q_f = qcount(by(rows, "full"), fab_v, False)
        n = sum(not r["in_scope"] for r in by(rows, "full"))
        res[tag] = {"nv": 100 * sum(q_nv.values()) / n, "full": 100 * sum(q_f.values()) / n, "test": sign(q_nv, q_f)}
    sens[name] = res
out["sensitivity"] = sens
summ = {}
for tag in ("1b", "3b", "squad"):
    vals = [v[tag] for v in sens.values()]
    summ[tag] = {"nv_min": min(x["nv"] for x in vals), "nv_max": max(x["nv"] for x in vals),
                 "full_min": min(x["full"] for x in vals), "full_max": max(x["full"] for x in vals),
                 "p_min": min(x["test"][2] for x in vals), "p_max": max(x["test"][2] for x in vals),
                 "n_variants": len(vals)}
out["sensitivity_summary"] = summ

Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("interaction", "holm_primary", "planned_3b", "variation", "squad_emf1",
                                      "squad_transitions_full", "poison_detail", "ollama_check")}, indent=1, default=str))
for k, v in T.items():
    print(f"{k:60s} {v}  holm_all={out['holm_all'][k]:.3f}")
