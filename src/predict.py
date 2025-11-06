import argparse, joblib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="models/xgb/xgb.pkl")
    ap.add_argument("--csv", required=True, help="feature rows to score")
    args = ap.parse_args()

    pack = joblib.load(args.pkl)
    df = pd.read_csv(args.csv)
    X = pack["scaler"].transform(df[pack["columns"]])
    proba = pack["model"].predict_proba(X)[:,1]
    print(pd.DataFrame({"proba_podium": proba}).head())

if __name__ == "__main__":
    main()
