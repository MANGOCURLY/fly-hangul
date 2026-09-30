import glob
import pyarrow.feather as feather
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

for path in sorted(glob.glob("*.feather")):
    t = feather.read_table(path, memory_map=True)
    print("=" * 80)
    print(path)
    print("rows:", t.num_rows)
    print(t.schema)
    print(t.slice(0, 5).to_pandas())
