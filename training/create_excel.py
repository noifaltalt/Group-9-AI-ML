import glob
import json
import pandas as pd

OUTPUT_XLSX = "data/SCT-data.csv"

dfs = []

for fp in glob.glob("data/**/SoccerTwos/*.json"):
    with open(fp,"r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data, sep=".")
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

final_df.to_csv(OUTPUT_XLSX, index=False)
print("Done:", OUTPUT_XLSX)

