"""3D 체험 페이지(docs/fly3d.html)용 데이터 내보내기

브라우저에서 실제 모델을 돌릴 수 있도록 다음을 저장한다 (docs/fly3d/):
- meta.json: 뉴런 위치(세포체 좌표, ORN만 그림용 배치), 역할, 부호, 사구체 번호, 자모 원형 패턴, 설정값, 표시용 연결
- w_connectome.bin / w_random.bin: 받는 뉴런 기준 CSR. indptr(int32, N+1) + pre(uint16, E) + weight(uint16, E)
- feats.bin: 통역사(점수표) 학습용 MBON 상태. 회로 2종 × 스텝 {2,4,6} × 자모 3,000개 × 97,
  뉴런(열)마다 최솟값~최댓값을 0~65535로 맞춘 uint16 (실제 커넥톰 MBON 값은 0~0.09로 좁아서 uint8이면 정보가 뭉개짐)
  라벨은 meta.json의 train_roles(직전 자모의 역할)
04_sequence.py의 회로·입력 규칙(v2 부호, 행 정규화, seed 0)을 그대로 쓴다.
"""
import importlib.util
import json
import os
import numpy as np
import pyarrow.feather as feather

spec = importlib.util.spec_from_file_location("seq", "04_sequence.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
OUT = "docs/fly3d"
os.makedirs(OUT, exist_ok=True)
N = S.N
ROLE_ORDER = ["ORN", "ALLN", "PN", "KC", "APL", "DPM", "MBON"]

# ---- 위치 ----
ann = feather.read_table("body-annotations-male-cns-v1.0-minconf-0.5.feather",
                         columns=["bodyId", "somaLocation", "rootSide"]).to_pandas().set_index("bodyId")
nodes = S.nodes
soma = ann.reindex(nodes["bodyId"])["somaLocation"].to_numpy()
side = ann.reindex(nodes["bodyId"])["rootSide"].fillna("unknown").to_numpy()
pos = np.full((N, 3), np.nan)
for i, v in enumerate(soma):
    if v is not None and len(v) == 3:
        pos[i] = v
known = ~np.isnan(pos[:, 0])
center = np.nanmean(pos[known], axis=0)
scale = np.nanmax(np.abs(pos[known] - center))
pos = (pos - center) / scale

role = S.role_arr
rng = np.random.default_rng(7)
brain_c = pos[known].mean(0)
# 좌표가 없는 비-ORN(몇 개)은 같은 역할 평균 근처에 둔다
for r in ROLE_ORDER:
    m = (role == r) & ~known & (r != "ORN")
    if m.any():
        pos[m] = np.nanmean(pos[(role == r) & known], axis=0) + rng.normal(0, 0.02, (m.sum(), 3))
# ORN: 좌우 ALLN 무리의 바깥쪽(더듬이 쪽)에 사구체별로 묶어서 배치 — 그림용 추정 위치
alln = role == "ALLN"
x_mid = brain_c[0]
orn_idx = np.flatnonzero(role == "ORN")
glom_of = {g: k for k, g in enumerate(S.glom_names)}
for k, i in enumerate(orn_idx):
    s = side[i] if side[i] in ("L", "R") else ("L" if k % 2 else "R")
    grp = alln & (np.sign(pos[:, 0] - x_mid) == (1 if s == "R" else -1))
    c = pos[grp].mean(0)
    out = (c - brain_c); out /= np.linalg.norm(out)
    g = glom_of.get(nodes.loc[i, "type"], 0)
    a = g / len(S.glom_names) * 2 * np.pi
    pos[i] = c + out * 0.28 + np.array([np.cos(a), np.sin(a), 0]) * 0.07 + rng.normal(0, 0.012, 3)

# ---- 연결 (v2 부호는 브라우저에서 sign 배열로 적용) ----
def write_csr(name, pre, post, w):
    order = np.lexsort((pre, post))
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(N + 1, np.int32)
    np.add.at(indptr, post + 1, 1)
    indptr = np.cumsum(indptr).astype(np.int32)
    with open(f"{OUT}/{name}", "wb") as fh:
        fh.write(indptr.tobytes()); fh.write(pre.astype(np.uint16).tobytes()); fh.write(np.minimum(w, 65535).astype(np.uint16).tobytes())
    return len(w)

E = write_csr("w_connectome.bin", S.pre, S.post, S.w.astype(np.int64))
g_rng = np.random.default_rng(1000 + 0)                      # 04_sequence.graphs(seed=0)와 같은 난수 순서
r_pre, r_post, r_w = g_rng.integers(0, N, len(S.w)), g_rng.integers(0, N, len(S.w)), g_rng.permutation(S.w)
write_csr("w_random.bin", r_pre, r_post, r_w.astype(np.int64))

# ---- 통역사 학습용 특징 ----
rng = np.random.default_rng(0)
P = S.prototypes(0)
seq = S.word_stream(rng, S.L, p_illegal=0.25)
ids = np.array([j for j, _ in seq]); roles = np.array([r for _, r in seq])
U = np.clip(P[ids] + rng.normal(0, S.SIGMA, (len(ids), S.G)).astype(np.float32), 0, None)
feats, lo, hi = [], [], []
Ws = S.graphs(0, True)
for gname in ["connectome", "random"]:
    for steps in [2, 4, 6]:
        S.STEPS_PER_JAMO = steps
        M, _ = S.run_stream(Ws[gname], U)
        a, b = M.min(0), M.max(0); b = np.where(b > a, b, a + 1e-9)
        feats.append(np.round((M - a) / (b - a) * 65535).astype(np.uint16)); lo.append(a.tolist()); hi.append(b.tolist())
        print(gname, steps, "done", flush=True)
np.stack(feats).tofile(f"{OUT}/feats.bin")

# ---- 표시용 연결: 주 경로 위주로 강한 것만 ----
wt = S.w
show = []
for a, b, k in [("ORN", "PN", 500), ("PN", "KC", 700), ("KC", "MBON", 700), ("ORN", "ALLN", 250), ("ALLN", "PN", 250), ("APL", "KC", 250), ("KC", "APL", 150), ("KC", "KC", 400)]:
    m = np.flatnonzero((role[S.pre] == a) & (role[S.post] == b))
    top = m[np.argsort(-wt[m])[:k]]
    show += [[int(S.pre[t]), int(S.post[t]), int(wt[t])] for t in top]

meta = {
    "source": "Janelia MaleCNS v1.0 (CC-BY 4.0)",
    "N": int(N), "E": int(E), "role_order": ROLE_ORDER,
    "role": [ROLE_ORDER.index(r) for r in role],
    "pos": np.round(pos, 4).tolist(),
    "sign": S.sign_of.astype(int).tolist(),
    "orn_glom": [int(glom_of.get(t, -1)) if r == "ORN" and isinstance(t, str) else -1 for r, t in zip(role, nodes["type"])],
    "mbon": S.mbon.tolist(),
    "glomeruli": S.glom_names,
    "jamo": S.JAMO,
    "prototypes": [np.flatnonzero(P[j]).tolist() for j in range(len(S.JAMO))],
    "settings": {"gain": S.GAIN, "leak": S.LEAK, "sigma": S.SIGMA, "steps": [2, 4, 6], "train_len": S.L,
                 "train_split": 0.7, "warmup": S.WARMUP},
    "feats_layout": {"circuits": ["connectome", "random"], "steps": [2, 4, 6], "T": S.L, "D": int(len(S.mbon)), "quant": "uint16 per-column min-max", "lo": lo, "hi": hi},
    "train_roles": roles.tolist(),
    "edges_show": show,
}
with open(f"{OUT}/meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, ensure_ascii=False, separators=(",", ":"))
for f in sorted(os.listdir(OUT)):
    print(f, round(os.path.getsize(f"{OUT}/{f}") / 1e6, 2), "MB")
