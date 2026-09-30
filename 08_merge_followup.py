"""병렬로 나눠 돌린 06_followup.py 결과(스텝 2 / 4 / 6 / 데이터 4배)를 docs/data/followup.json 하나로 합친다."""
import json

out = json.load(open("data/followup_steps2.json", encoding="utf-8"))
for part in ["4", "6", "e1"]:
    p = json.load(open(f"data/followup_part_{part}.json", encoding="utf-8"))
    out["E3_steps"].update(p["E3_steps"])
    out["E1_data"].update(p["E1_data"])
with open("docs/data/followup.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False)
print("steps:", sorted(out["E3_steps"]), "E1:", sorted(out["E1_data"]))
