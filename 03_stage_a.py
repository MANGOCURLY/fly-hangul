"""단계 A: 자모 40개 알아보기 (저장소 컴퓨팅, v1 = 부호 없음)

구조: 자모 → 사구체 53종 입력 패턴(+노이즈) → ORN → 회로 T스텝 → MBON 97개 상태 → 선형 분류기(ridge)
비교군: 실제 커넥톰 / 랜덤 그래프(같은 뉴런 수·연결 수·가중치) / 셔플(각 뉴런의 입력·출력 연결 수 보존) / 입력만(회로 없음)
출력: docs/data/stage_a.json
"""
import json
import sys
import time
import numpy as np
import scipy.sparse as sp
import pyarrow.feather as feather

JAMO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ" "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
ACTIVE = 8            # 자모 하나당 켜지는 사구체 수 (53개 중)
T = 8                 # 회로를 돌리는 스텝 수 (ORN→PN→KC→MBON 깊이 3 + 되먹임 여유)
GAIN = 0.9            # 행 정규화 후 되먹임 세기 (스펙트럴 반경 ≤ 0.9)
LEAK = 0.5            # 상태 갱신 비율
N_TRAIN, N_TEST = 30, 20
NOISES = [0.5, 1.0, 1.5, 2.0, 3.0]
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [1e-3, 1e-2, 1e-1, 1, 10]
QUICK = "--quick" in sys.argv   # 시간 측정용: 노이즈 1개, seed 1개

nodes = feather.read_feather("data/nodes.feather")
edges = feather.read_feather("data/edges.feather")
N = len(nodes)
idx = {b: i for i, b in enumerate(nodes["bodyId"])}
pre = edges["body_pre"].map(idx).to_numpy()
post = edges["body_post"].map(idx).to_numpy()
w = edges["weight"].to_numpy().astype(np.float32)

# 사구체 타입이 비어 있는 ORN 4개는 회로에는 남기되 입력은 넣지 않는다
orn = np.flatnonzero((nodes["role"].to_numpy() == "ORN") & nodes["type"].notna().to_numpy())
glom_names = sorted(nodes.loc[orn, "type"].unique())
glom_of_orn = nodes.loc[orn, "type"].map({g: k for k, g in enumerate(glom_names)}).to_numpy()
mbon = np.flatnonzero(nodes["role"].to_numpy() == "MBON")
G = len(glom_names)


def build_W(pre, post, w):
    """행(받는 뉴런)마다 입력 합이 1이 되게 정규화 → 모든 그래프에 같은 규칙 적용."""
    W = sp.csr_matrix((w, (post, pre)), shape=(N, N), dtype=np.float32)
    rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1
    return (sp.diags(1 / rs) @ W * GAIN).tocsr().astype(np.float32)


def graphs(seed):
    rng = np.random.default_rng(1000 + seed)
    return {
        "connectome": build_W(pre, post, w),
        "random": build_W(rng.integers(0, N, len(w)), rng.integers(0, N, len(w)), rng.permutation(w)),
        "shuffle": build_W(pre, rng.permutation(post), w),
    }


def run(W, U):
    """U: (G, S) 사구체 입력 → MBON 최종 상태 (S, 97)."""
    drive = np.zeros((N, U.shape[1]), dtype=np.float32)
    drive[orn] = U[glom_of_orn]
    x = np.zeros_like(drive)
    for _ in range(T):
        x = (1 - LEAK) * x + LEAK * np.tanh(W @ x + drive)
    return x[mbon].T


def ridge_acc(Xtr, ytr, Xte, yte, rng):
    """학습 데이터 안에서 20%를 떼어 λ를 고른 뒤, 전체 학습 데이터로 다시 맞추고 테스트."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = np.eye(len(JAMO))[ytr]
    def fit(X, Y, lam):
        Xb = np.hstack([X, np.ones((len(X), 1))])
        return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    def pred(X, B):
        return (np.hstack([X, np.ones((len(X), 1))]) @ B).argmax(1)
    perm = rng.permutation(len(Xtr)); cut = int(len(Xtr) * 0.8)
    a, b = perm[:cut], perm[cut:]
    best = max(LAMBDAS, key=lambda l: (pred(Xtr[b], fit(Xtr[a], Y[a], l)) == ytr[b]).mean())
    return float((pred(Xte, fit(Xtr, Y, best)) == yte).mean())


results = {c: {str(s): [] for s in NOISES} for c in ["connectome", "random", "shuffle", "input_only"]}
panel2 = None
noises = NOISES[:1] if QUICK else NOISES
seeds = SEEDS[:1] if QUICK else SEEDS
t0 = time.time()
for seed in seeds:
    rng = np.random.default_rng(seed)
    proto = np.zeros((len(JAMO), G), dtype=np.float32)
    for j in range(len(JAMO)):
        proto[j, rng.choice(G, ACTIVE, replace=False)] = 1
    Ws = graphs(seed)
    for sigma in noises:
        per = N_TRAIN + N_TEST
        y = np.repeat(np.arange(len(JAMO)), per)
        U = proto[y] + rng.normal(0, sigma, (len(y), G)).astype(np.float32)
        U = np.clip(U, 0, None)
        is_tr = np.tile(np.r_[np.ones(N_TRAIN, bool), np.zeros(N_TEST, bool)], len(JAMO))
        feats = {"input_only": U}
        for name, W in Ws.items():
            feats[name] = run(W, U.T)
        for name, F in feats.items():
            acc = ridge_acc(F[is_tr], y[is_tr], F[~is_tr], y[~is_tr], np.random.default_rng(seed))
            results[name][str(sigma)].append(acc)
        if seed == 0 and sigma == 0.5:
            M = feats["connectome"]
            panel2 = [M[y == j].mean(0).round(4).tolist() for j in range(len(JAMO))]
        print(f"seed {seed} σ={sigma}: " + ", ".join(f"{k}={v[str(sigma)][-1]:.3f}" for k, v in results.items()),
              f"({time.time()-t0:.0f}s)", flush=True)

if not QUICK:
    out = {
        "task": "자모 40개 분류", "chance": 1 / len(JAMO), "noises": NOISES, "seeds": SEEDS,
        "settings": {"active_glomeruli": ACTIVE, "steps": T, "gain": GAIN, "leak": LEAK,
                     "train_per_jamo": N_TRAIN, "test_per_jamo": N_TEST, "signs": "없음 (v1)"},
        "accuracy": results,
        "jamo": JAMO,
        "mbon_types": nodes.loc[mbon, "type"].tolist(),
        "mbon_response_sigma0.5": panel2,
    }
    with open("docs/data/stage_a.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("saved docs/data/stage_a.json")
