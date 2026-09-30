"""대시보드 패널 ①(회로 지도)용 요약 JSON을 만든다.
입력: data/nodes.feather, data/edges.feather  출력: docs/data/circuit.json
"""
import json
import pyarrow.feather as feather

ORDER = ["ORN", "ALLN", "PN", "KC", "APL", "DPM", "MBON"]
MAIN_PATH = {("ORN", "PN"), ("PN", "KC"), ("KC", "MBON")}

n = feather.read_feather("data/nodes.feather")
e = feather.read_feather("data/edges.feather")
role = n.set_index("bodyId")["role"]
e["pre"] = e["body_pre"].map(role)
e["post"] = e["body_post"].map(role)

nodes = []
for r in ORDER:
    s = n[n["role"] == r]
    nodes.append({
        "role": r,
        "neurons": int(len(s)),
        "types": int(s["type"].nunique()),
        "nt": {k: int(v) for k, v in s["consensus_nt"].value_counts().items()},
    })

g = e.groupby(["pre", "post"])["weight"].agg(["size", "sum"]).reset_index()
links = [{
    "pre": row.pre, "post": row.post,
    "connections": int(row.size), "synapses": int(row.sum),
    "main": (row.pre, row.post) in MAIN_PATH,
} for row in g.itertuples()]

out = {
    "source": "Janelia MaleCNS v1.0 (CC-BY 4.0), connectome-weights minconf 0.5, status=Traced",
    "total_neurons": int(len(n)), "total_connections": int(len(e)), "total_synapses": int(e["weight"].sum()),
    "nodes": nodes, "links": links,
}
with open("docs/data/circuit.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(len(links), "links;", out["total_synapses"], "synapses")
