"""1단계: 후각-버섯체 부분회로(ORN, ALLN, ALPN, KC, APL, DPM, MBON)만 뽑아 저장한다.
출력: data/nodes.feather (뉴런 목록), data/edges.feather (부분회로 내부 연결)
"""
import time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc

CLASSES = {"olfactory": "ORN", "ALLN": "ALLN", "ALPN": "PN", "Kenyon_Cell": "KC", "MBON": "MBON"}
TYPES = {"APL": "APL", "DPM": "DPM"}

ann = feather.read_table("body-annotations-male-cns-v1.0-minconf-0.5.feather",
                         columns=["bodyId", "type", "class", "somaSide", "status"]).to_pandas()
ann = ann[ann["status"] == "Traced"]
role = ann["class"].map(CLASSES)
role = role.fillna(ann["type"].map(TYPES))
nodes = ann.assign(role=role).dropna(subset=["role"])[["bodyId", "role", "type", "somaSide"]]

nt = feather.read_table("body-neurotransmitters-male-cns-v1.0.feather",
                        columns=["body", "consensus_nt", "predicted_nt_confidence"]).to_pandas()
nodes = nodes.merge(nt.rename(columns={"body": "bodyId"}), on="bodyId", how="left")
nodes = nodes.reset_index(drop=True)
print(nodes["role"].value_counts().to_dict())

ids = pa.array(nodes["bodyId"].to_numpy())
t0 = time.time()
kept, seen = [], 0
with pa.memory_map("connectome-weights-male-cns-v1.0-minconf-0.5.feather") as src:
    reader = ipc.open_file(src)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        seen += b.num_rows
        mask = pc.and_(pc.is_in(b["body_pre"], value_set=ids), pc.is_in(b["body_post"], value_set=ids))
        kept.append(b.filter(mask))
edges = pa.Table.from_batches(kept)
print(f"scanned {seen:,} rows, kept {edges.num_rows:,} in {time.time()-t0:.0f}s")

feather.write_feather(nodes, "data/nodes.feather")
feather.write_feather(edges, "data/edges.feather")
