"""사전 등록한 추가 실험 (docs/data/followup_plan.json)

E3: 자모당 스텝 2 / 4 / 6 → B0, B, C
E1: 스트림 3,000 → 12,000 자모 (스텝 2) → B, C
모두 v2(부호 있음), seed 0~4, 처음 설계(지연 없이 읽기). 04_sequence.py의 함수를 그대로 쓴다.
출력: docs/data/followup.json (실험이 하나 끝날 때마다 저장)
"""
import importlib.util
import json
import time
import numpy as np

spec = importlib.util.spec_from_file_location("seq", "04_sequence.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

CONDS = ["connectome", "random", "shuffle", "input_only", "ideal_memory"]
KS = [0, 1, 2, 3, 4, 5]
OUT = "docs/data/followup.json"
out = {"plan": "docs/data/followup_plan.json", "E3_steps": {}, "E1_data": {}}


def save():
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)


def run_tasks(seed, steps, L, with_b0):
    """04_sequence.py 본 실험과 같은 순서로 난수를 쓴다(스텝 2·L 3,000이면 본 실험을 그대로 재현)."""
    S.STEPS_PER_JAMO = steps
    rng = np.random.default_rng(seed)
    P = S.prototypes(seed)
    Ws = S.graphs(seed, True)
    noisy = lambda ids: np.clip(P[ids] + rng.normal(0, S.SIGMA, (len(ids), S.G)).astype(np.float32), 0, None)
    res = {}
    ids = rng.integers(0, len(S.JAMO), L)
    U0 = noisy(ids)
    if with_b0:
        F = S.features(U0, L, Ws)
        res["B0"] = {}
        for k in KS:
            y = np.roll(ids, k); m = np.ones(L, bool); m[:k] = False
            res["B0"][str(k)] = S.evaluate(F, y, m, len(S.JAMO), seed, "plain")
    seq = S.word_stream(rng, L, p_illegal=0.25)
    ids = np.array([j for j, _ in seq]); roles = np.array([r for _, r in seq])
    F = S.features(noisy(ids), L, Ws)
    yB = np.roll(roles, 1); mB = np.ones(L, bool); mB[0] = False
    res["B"] = S.evaluate(F, yB, mB, 3, seed, "balanced")
    mC = (roles == 1) & (np.roll(roles, 1) == 2); mC[0] = False
    yC = np.array([int(p in S.ILLEGAL) for p in np.roll(ids, 1)])
    res["C"] = S.evaluate(F, yC, mC, 2, seed, "balanced")
    return res


def collect(store, key, seed, r):
    d = store.setdefault(key, {})
    for task, v in r.items():
        if task == "B0":
            for k, rr in v.items():
                for c in CONDS: d.setdefault("B0", {}).setdefault(k, {}).setdefault(c, []).append(rr[c])
        else:
            for c in CONDS: d.setdefault(task, {}).setdefault(c, []).append(v[c])


# 병렬 실행용: --part 4 | 6 | e1 (없으면 전부 순서대로). 부분 결과는 data/followup_part_<part>.json
import sys
part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else None
if part:
    OUT = f"data/followup_part_{part}.json"
t0 = time.time()
for steps in [2, 4, 6]:
    if part and part != str(steps):
        continue
    for seed in S.SEEDS:
        r = run_tasks(seed, steps, 3000, with_b0=True)
        collect(out["E3_steps"], str(steps), seed, r)
        print(f"E3 steps={steps} seed={seed}: B conn={r['B']['connectome']:.3f} rand={r['B']['random']:.3f} shuf={r['B']['shuffle']:.3f}"
              f" | C conn={r['C']['connectome']:.3f} rand={r['C']['random']:.3f} shuf={r['C']['shuffle']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    save()

if part in (None, "e1"):
    for seed in S.SEEDS:
        r = run_tasks(seed, 2, 12000, with_b0=False)
        collect(out["E1_data"], "12000", seed, r)
        print(f"E1 L=12000 seed={seed}: B conn={r['B']['connectome']:.3f} rand={r['B']['random']:.3f} shuf={r['B']['shuffle']:.3f}"
              f" | C conn={r['C']['connectome']:.3f} rand={r['C']['random']:.3f} shuf={r['C']['shuffle']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    save()
print("saved", OUT)
