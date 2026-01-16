import glob
import json
import pandas as pd
import os

OUTPUT_XLSX = "data/SCT-data.csv"

def standardize_run_id(data,fp):
    hostname = os.path.basename(os.path.dirname(os.path.dirname(fp)))
    old_id = data["run_id"]
    id = hostname + "_" + old_id
    data["run_id"] = id

def standardize_score(data):
    old = data["efficiency_score"]
    data["efficiency_score"] = old*120

def main():
    dfs = []
    for fp in glob.glob("data/**/SoccerTwos/*.json"):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        standardize_run_id(data,fp)
        standardize_score(data)
        df = pd.json_normalize(data, sep=".")
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    final_df.to_csv(OUTPUT_XLSX, index=False)
    print("Done:", OUTPUT_XLSX)

if __name__ == "__main__":
    main()

