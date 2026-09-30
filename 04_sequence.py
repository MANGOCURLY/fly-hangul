"""단계 B0 / B / C: 자모를 하나씩 순서대로 넣는 과제 (기억이 필요한 과제)

B0 기억력: 지금 MBON 상태로 k(0~5)단계 전 자모를 맞힌다. 40지선다, 찍기 2.5%.
B  받침/첫소리: 지금 MBON 상태로 '직전 자모'가 모음/첫소리/받침 중 무엇이었는지 맞힌다.
   직전 자모의 역할은 '그 다음(=지금) 자모가 모음인가'로 정해지므로, 직전과 지금을 함께 봐야 풀린다.
C  받침 규칙: 자음 뒤에 자음이 온 자리에서, 직전 받침이 ㄸ·ㅃ·ㅉ(받침이 될 수 없음)인지 맞힌다.

비교군: connectome / random / shuffle (MBON 97개를 읽음), input_only(지금 입력만),
        ideal_memory(지금 입력 + 직전 입력을 그대로 붙인 것 = 기억이 완벽할 때의 참고 상한)
v1 = 부호 없음, v2 = 신경전달물질 예측으로 부호 부여 (GABA·글루탐산 = 억제, 그 외 = 흥분)
출력: docs/data/stage_b.json
"""
import json
import time
import numpy as np
import scipy.sparse as sp
import pyarrow.feather as feather

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄱㅅ", "ㄴ", "ㄴㅈ", "ㄴㅎ", "ㄷ", "ㄹ", "ㄹㄱ", "ㄹㅁ", "ㄹㅂ", "ㄹㅅ", "ㄹㅌ",
        "ㄹㅍ", "ㄹㅎ", "ㅁ", "ㅂ", "ㅂㅅ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JAMO = list(CHO + JUNG)
JID = {j: i for i, j in enumerate(JAMO)}
IS_VOWEL = np.array([j in JUNG for j in JAMO])
ROLES = ["모음", "첫소리", "받침"]
ILLEGAL = {JID["ㄸ"], JID["ㅃ"], JID["ㅉ"]}

WORDS = ("한글 한국어 초파리 사랑 사람 학교 선생님 학생 공부 친구 가족 엄마 아빠 동생 언니 오빠 누나 형 "
         "물 불 집 밥 빵 꽃 닭 흙 값 삶 앉다 많다 읽다 없다 넓다 여덟 괜찮아 고맙습니다 안녕하세요 감사합니다 "
         "미안해요 좋아요 싫어요 먹다 마시다 가다 오다 보다 듣다 말하다 쓰다 읽기 쓰기 노래 음악 영화 책 "
         "커피 우유 김치 불고기 비빔밥 떡볶이 라면 과일 사과 바나나 딸기 수박 포도 하늘 바다 산 강 나무 "
         "바람 구름 비 눈 봄 여름 가을 겨울 아침 점심 저녁 오늘 내일 어제 시간 시계 전화 컴퓨터 버스 "
         "지하철 기차 비행기 병원 약국 시장 은행 우체국 도서관 공원 식당 카페 호텔 방 문 창문 의자 책상 "
         "옷 신발 모자 가방 돈 이름 나라 도시 프랑스 서울 보르도 날씨 행복 기분 마음 생각 꿈 빛 밖 부엌 "
         "깎다 볶음 씻다 짧다 찌개 뽀뽀 빨리 똑똑 쪽지 까치 꼬리 뿌리 찐빵 쌀 씨앗").split()


def decompose(word):
    """단어 → [(자모 번호, 역할 번호)]. 역할: 0 모음, 1 첫소리, 2 받침. 겹받침은 두 자음으로 나눈다."""
    out = []
    for ch in word:
        s = ord(ch) - 0xAC00
        out.append((JID[CHO[s // 588]], 1))
        out.append((JID[JUNG[(s % 588) // 28]], 0))
        for c in JONG[s % 28]:
            out.append((JID[c], 2))
    return out


def illegal_syllable(rng):
    """ㄸ·ㅃ·ㅉ 받침을 가진 (규칙에 어긋난) 가짜 음절."""
    return [(JID[CHO[rng.integers(19)]], 1), (JID[JUNG[rng.integers(21)]], 0), (int(rng.choice(sorted(ILLEGAL))), 2)]


def word_stream(rng, n, p_illegal):
    seq = []
    while len(seq) < n:
        if rng.random() < p_illegal:
            seq += illegal_syllable(rng)
        seq += decompose(WORDS[rng.integers(len(WORDS))])
    return seq[:n]


# ---------------- 회로 ----------------
ACTIVE, STEPS_PER_JAMO, GAIN, LEAK, SIGMA = 8, 2, 0.9, 0.5, 0.5
L = 3000            # 스트림 길이(자모 수). 앞 70% 학습, 뒤 30% 테스트
WARMUP = 20
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [1e-3, 1e-2, 1e-1, 1, 10, 100]
INHIB = {"gaba", "glutamate"}

nodes = feather.read_feather("data/nodes.feather")
edges = feather.read_feather("data/edges.feather")
N = len(nodes)
idx = {b: i for i, b in enumerate(nodes["bodyId"])}
pre = edges["body_pre"].map(idx).to_numpy()
post = edges["body_post"].map(idx).to_numpy()
w = edges["weight"].to_numpy().astype(np.float32)
sign_of = np.where(nodes["consensus_nt"].isin(INHIB).to_numpy(), -1.0, 1.0).astype(np.float32)
role_arr = nodes["role"].to_numpy()
orn = np.flatnonzero((role_arr == "ORN") & nodes["type"].notna().to_numpy())
glom_names = sorted(nodes.loc[orn, "type"].unique())
glom_of_orn = nodes.loc[orn, "type"].map({g: k for k, g in enumerate(glom_names)}).to_numpy()
mbon = np.flatnonzero(role_arr == "MBON")
G = len(glom_names)
ROLE_ORDER = ["ORN", "ALLN", "PN", "KC", "APL", "DPM", "MBON"]
role_idx = {r: np.flatnonzero(role_arr == r) for r in ROLE_ORDER}


def build_W(pre, post, w, signed):
    """행마다 |입력| 합이 1이 되게 정규화 → 모든 그래프·버전에 같은 규칙. 부호는 보내는 뉴런이 정한다(데일의 법칙)."""
    v = w * (sign_of[pre] if signed else 1)
    W = sp.csr_matrix((v, (post, pre)), shape=(N, N), dtype=np.float32)
    rs = np.asarray(abs(W).sum(axis=1)).ravel()
    rs[rs == 0] = 1
    return (sp.diags(1 / rs) @ W * GAIN).tocsr().astype(np.float32)


def graphs(seed, signed):
    rng = np.random.default_rng(1000 + seed)
    return {
        "connectome": build_W(pre, post, w, signed),
        "random": build_W(rng.integers(0, N, len(w)), rng.integers(0, N, len(w)), rng.permutation(w), signed),
        "shuffle": build_W(pre, rng.permutation(post), w, signed),
    }


def prototypes(seed):
    rng = np.random.default_rng(seed)
    P = np.zeros((len(JAMO), G), dtype=np.float32)
    for j in range(len(JAMO)):
        P[j, rng.choice(G, ACTIVE, replace=False)] = 1
    return P


def run_stream(W, U, record_roles=False):
    """U: (L, G) 자모별 입력. 각 자모를 STEPS_PER_JAMO 스텝 유지하고 마지막 상태를 기록."""
    x = np.zeros(N, dtype=np.float32)
    out = np.zeros((len(U), len(mbon)), dtype=np.float32)
    roles = np.zeros((len(U), len(ROLE_ORDER)), dtype=np.float32) if record_roles else None
    drive = np.zeros(N, dtype=np.float32)
    for t, u in enumerate(U):
        drive[orn] = u[glom_of_orn]
        for _ in range(STEPS_PER_JAMO):
            x = (1 - LEAK) * x + LEAK * np.tanh(W @ x + drive)
        out[t] = x[mbon]
        if record_roles:
            roles[t] = [np.abs(x[role_idx[r]]).mean() for r in ROLE_ORDER]
    return out, roles


def fit_ridge(X, y, k, rng):
    """클래스 수가 달라도 공평하게(클래스별 가중치), 학습 데이터 20%로 λ를 고른다. 반환: 예측 함수."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    cnt = np.bincount(y, minlength=k).astype(float)
    sw = np.sqrt(1 / np.maximum(cnt[y], 1))[:, None]
    Y = np.eye(k)[y]
    def solve(Z, Y, sw, lam):
        A = Z * sw
        return np.linalg.solve(A.T @ A + lam * np.eye(Z.shape[1]), A.T @ (Y * sw))
    perm = rng.permutation(len(Z)); cut = int(len(Z) * 0.8); a, b = perm[:cut], perm[cut:]
    best = max(LAMBDAS, key=lambda l: bal_acc((Z[b] @ solve(Z[a], Y[a], sw[a], l)).argmax(1), y[b], k))
    B = solve(Z, Y, sw, best)
    return lambda Xn: (np.hstack([(Xn - mu) / sd, np.ones((len(Xn), 1))]) @ B)


def bal_acc(pred, y, k):
    accs = [(pred[y == c] == c).mean() for c in range(k) if (y == c).any()]
    return float(np.mean(accs))


def features(U, seq_len, Ws):
    prev = np.vstack([np.zeros((1, G), np.float32), U[:-1]])
    F = {"input_only": U, "ideal_memory": np.hstack([U, prev])}
    for name, W in Ws.items():
        F[name] = run_stream(W, U)[0]
    return F


def evaluate(F, y, mask, k, seed, metric):
    tr = np.zeros(len(y), bool); tr[WARMUP:int(len(y) * 0.7)] = True
    te = np.zeros(len(y), bool); te[int(len(y) * 0.7):] = True
    tr &= mask; te &= mask
    res = {}
    for name, X in F.items():
        f = fit_ridge(X[tr], y[tr], k, np.random.default_rng(seed))
        p = f(X[te]).argmax(1)
        res[name] = bal_acc(p, y[te], k) if metric == "balanced" else float((p == y[te]).mean())
    return res


if __name__ == "__main__":
    CONDS = ["connectome", "random", "shuffle", "input_only", "ideal_memory"]
    KS = [0, 1, 2, 3, 4, 5]
    res = {v: {"B0": {str(k): {c: [] for c in CONDS} for k in KS},
               "B": {c: [] for c in CONDS}, "C": {c: [] for c in CONDS}} for v in ["v1", "v2"]}
    t0 = time.time()
    for version, signed in [("v1", False), ("v2", True)]:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            P = prototypes(seed)
            Ws = graphs(seed, signed)
            noisy = lambda ids: np.clip(P[ids] + rng.normal(0, SIGMA, (len(ids), G)).astype(np.float32), 0, None)

            # B0: 무작위 자모 스트림
            ids = rng.integers(0, len(JAMO), L)
            F = features(noisy(ids), L, Ws)
            for k in KS:
                y = np.roll(ids, k); m = np.ones(L, bool); m[:k] = False
                r = evaluate(F, y, m, len(JAMO), seed, "plain")
                for c in CONDS: res[version]["B0"][str(k)][c].append(r[c])

            # B, C: 실제 단어 스트림 (+ ㄸ·ㅃ·ㅉ 받침 가짜 음절 섞기)
            seq = word_stream(rng, L, p_illegal=0.25)
            ids = np.array([j for j, _ in seq]); roles = np.array([r for _, r in seq])
            F = features(noisy(ids), L, Ws)
            yB = np.roll(roles, 1); mB = np.ones(L, bool); mB[0] = False
            r = evaluate(F, yB, mB, 3, seed, "balanced")
            for c in CONDS: res[version]["B"][c].append(r[c])
            prev_ids = np.roll(ids, 1)
            mC = (roles == 1) & (np.roll(roles, 1) == 2); mC[0] = False    # 받침 바로 뒤 첫소리 자리
            yC = np.array([int(p in ILLEGAL) for p in prev_ids])
            r = evaluate(F, yC, mC, 2, seed, "balanced")
            for c in CONDS: res[version]["C"][c].append(r[c])

            print(f"{version} seed {seed}: B0(k=1) " + " ".join(f"{c}={res[version]['B0']['1'][c][-1]:.3f}" for c in CONDS)
                  + " | B " + " ".join(f"{c}={res[version]['B'][c][-1]:.3f}" for c in CONDS)
                  + " | C " + " ".join(f"{c}={res[version]['C'][c][-1]:.3f}" for c in CONDS)
                  + f" ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 스토리/재생용 기록: 실제 커넥톰, seed 0 ----------------
    mean = lambda v: float(np.mean(v))
    best_v = max(["v1", "v2"], key=lambda v: mean(res[v]["B"]["connectome"]))
    rng = np.random.default_rng(0)
    P = prototypes(0)
    W = graphs(0, best_v == "v2")["connectome"]
    seq = word_stream(rng, L, p_illegal=0.25)
    demo_words = ["한글", "초파리", "닭", "안녕하세요"]
    demo = [x for wd in demo_words for x in decompose(wd)]
    full = seq + demo
    ids = np.array([j for j, _ in full]); roles = np.array([r for _, r in full])
    U = np.clip(P[ids] + rng.normal(0, SIGMA, (len(ids), G)).astype(np.float32), 0, None)
    M, R = run_stream(W, U, record_roles=True)
    yB = np.roll(roles, 1)
    tr = np.arange(WARMUP, int(L * 0.7))
    f = fit_ridge(M[tr], yB[tr], 3, np.random.default_rng(0))
    d0 = len(seq)
    scores = f(M[d0:])
    trace = []
    for i in range(len(demo)):
        t = d0 + i
        trace.append({
            "jamo": JAMO[ids[t]], "role": int(roles[t]),
            "prev_pred": int(scores[i].argmax()), "prev_true": int(yB[t]),
            "glom": np.flatnonzero(P[ids[t]]).tolist(),
            "roles_act": R[t].round(4).tolist(),
            "mbon": M[t].round(3).tolist(),
        })

    out = {
        "settings": {"steps_per_jamo": STEPS_PER_JAMO, "gain": GAIN, "leak": LEAK, "sigma": SIGMA, "stream_len": L,
                     "seeds": SEEDS, "illegal_rate": 0.25, "inhibitory_nt": sorted(INHIB)},
        "roles": ROLES, "conds": CONDS, "ks": KS, "results": res,
        "demo": {"version": best_v, "words": demo_words, "role_order": ROLE_ORDER, "trace": trace},
        "prototypes_seed0": {JAMO[j]: np.flatnonzero(P[j]).tolist() for j in range(len(JAMO))},
        "glomeruli": glom_names,
    }
    with open("docs/data/stage_b.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print("saved docs/data/stage_b.json; demo version", best_v)
