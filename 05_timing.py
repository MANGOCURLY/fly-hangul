"""추가 분석 (단계 B0 결과를 본 뒤 설계): 읽는 시점을 회로마다 공정하게 맞춰 주기

B0에서 실제 커넥톰은 지금 넣은 자모가 출력(MBON)에 3단계 늦게 도착했다(지금 2.5%, 3개 전 43%).
그래서 B·C를 '직전 자모를 넣은 직후'에만 읽으면 실제 회로에 불리하다.
여기서는 회로(connectome/random/shuffle)마다 d = 0~3 자모 늦게 읽는 것을 허용하고,
d는 학습 구간 안의 검증 데이터로만 고른다(테스트 데이터는 d 선택에 쓰지 않음).
input_only / ideal_memory는 늦게 읽으면 미래 입력을 보게 되므로 d = 0 고정.
주의: 늦게 읽으면 회로는 그 사이 들어온 다음 자모도 본다. 단어 통계가 조금 새어 들어갈 수 있다.
v2(부호 있음)만 실행. 출력: docs/data/stage_b_timing.json (+ 재생용 trace 갱신)
"""
import importlib.util
import json
import numpy as np

spec = importlib.util.spec_from_file_location("seq", "04_sequence.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

DELAYS = [0, 1, 2, 3]
READERS = ["connectome", "random", "shuffle"]


def split_idx(L):
    tr_end, va_start = int(L * 0.7), int(L * 0.56)
    return tr_end, va_start


def shifted(F, d):
    """시점 t의 라벨을 F[t+d]로 읽는다. 끝의 d개는 읽을 수 없으므로 0으로 채우고 마스크로 뺀다."""
    return np.vstack([F[d:], np.zeros((d, F.shape[1]), F.dtype)]) if d else F


def eval_delay(F, y, mask, k, seed):
    L = len(y); tr_end, va_start = split_idx(L)
    res, chosen = {}, {}
    for name, X in F.items():
        ds = DELAYS if name in READERS else [0]
        def score(d, fit_rng, tr_mask, te_mask):
            Xd = shifted(X, d); valid = mask.copy(); valid[L - d:] = False
            f = S.fit_ridge(Xd[tr_mask & valid], y[tr_mask & valid], k, fit_rng)
            m = te_mask & valid
            return S.bal_acc(f(Xd[m]).argmax(1), y[m], k)
        base = np.zeros(L, bool)
        tr_a = base.copy(); tr_a[S.WARMUP:va_start] = True
        va = base.copy(); va[va_start:tr_end] = True
        best = max(ds, key=lambda d: score(d, np.random.default_rng(seed), tr_a, va))
        tr = base.copy(); tr[S.WARMUP:tr_end] = True
        te = base.copy(); te[tr_end:] = True
        res[name] = score(best, np.random.default_rng(seed), tr, te)
        chosen[name] = best
    return res, chosen


CONDS = ["connectome", "random", "shuffle", "input_only", "ideal_memory"]
res = {t: {c: [] for c in CONDS} for t in ["B", "C"]}
delays = {t: {c: [] for c in CONDS} for t in ["B", "C"]}
for seed in S.SEEDS:
    rng = np.random.default_rng(seed)
    P = S.prototypes(seed)
    Ws = S.graphs(seed, True)
    rng.integers(0, len(S.JAMO), S.L)                      # B0와 같은 난수 흐름을 유지하려고 한 번 소비
    _ = rng.normal(0, S.SIGMA, (S.L, S.G))
    seq = S.word_stream(rng, S.L, p_illegal=0.25)
    ids = np.array([j for j, _ in seq]); roles = np.array([r for _, r in seq])
    U = np.clip(P[ids] + rng.normal(0, S.SIGMA, (len(ids), S.G)).astype(np.float32), 0, None)
    F = S.features(U, S.L, Ws)
    yB = np.roll(roles, 1); mB = np.ones(S.L, bool); mB[0] = False
    r, d = eval_delay(F, yB, mB, 3, seed)
    for c in CONDS: res["B"][c].append(r[c]); delays["B"][c].append(d[c])
    mC = (roles == 1) & (np.roll(roles, 1) == 2); mC[0] = False
    yC = np.array([int(p in S.ILLEGAL) for p in np.roll(ids, 1)])
    r, d = eval_delay(F, yC, mC, 2, seed)
    for c in CONDS: res["C"][c].append(r[c]); delays["C"][c].append(d[c])
    print(f"seed {seed}: B " + " ".join(f"{c}={res['B'][c][-1]:.3f}(d{delays['B'][c][-1]})" for c in CONDS)
          + " | C " + " ".join(f"{c}={res['C'][c][-1]:.3f}(d{delays['C'][c][-1]})" for c in CONDS), flush=True)

# 재생용: 실제 커넥톰, seed 0, B에서 가장 많이 고른 지연으로 판정
d_demo = int(np.bincount(delays["B"]["connectome"]).argmax())
rng = np.random.default_rng(0)
P = S.prototypes(0)
W = S.graphs(0, True)["connectome"]
seq = S.word_stream(rng, S.L, p_illegal=0.25)
demo_words = ["한글", "초파리", "닭", "안녕하세요"]
demo = [x for wd in demo_words for x in S.decompose(wd)]
tail = S.decompose("사랑")                                     # 마지막 자모들도 d만큼 뒤에 읽을 수 있게 덧붙임
full = seq + demo + tail
ids = np.array([j for j, _ in full]); roles = np.array([r for _, r in full])
U = np.clip(P[ids] + rng.normal(0, S.SIGMA, (len(ids), S.G)).astype(np.float32), 0, None)
M, R = S.run_stream(W, U, record_roles=True)
yB = np.roll(roles, 1)
Md = shifted(M, d_demo)
tr = np.arange(S.WARMUP, int(S.L * 0.7))
f = S.fit_ridge(Md[tr], yB[tr], 3, np.random.default_rng(0))
d0 = len(seq)
trace = []
for i in range(len(demo)):
    t = d0 + i
    trace.append({"jamo": S.JAMO[ids[t]], "role": int(roles[t]),
                  "prev_pred": int(f(Md[t:t + 1]).argmax()), "prev_true": int(yB[t]),
                  "glom": np.flatnonzero(P[ids[t]]).tolist(),
                  "roles_act": R[t].round(4).tolist(), "mbon": M[t].round(3).tolist()})

out = {"version": "v2", "delays_tried": DELAYS, "results": res, "chosen_delay": delays,
       "demo": {"delay": d_demo, "words": demo_words, "role_order": S.ROLE_ORDER, "trace": trace}}
with open("docs/data/stage_b_timing.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False)
print("saved docs/data/stage_b_timing.json; demo delay", d_demo,
      "demo correct", sum(x["prev_pred"] == x["prev_true"] for x in trace[1:]), "/", len(trace) - 1)
