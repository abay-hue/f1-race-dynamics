import argparse, os, joblib, pandas as pd, numpy as np, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/f1_features.csv")
    ap.add_argument("--out", default="models/xgb")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    y = df["podium"]  # 0/1
    X = df.drop(columns=["podium"])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=4
    )
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    print(f"AUC: {auc:.4f}")

    joblib.dump({"model": model, "scaler": scaler, "columns": X.columns.tolist()}, os.path.join(args.out, "xgb.pkl"))
    print("✅ Saved:", os.path.join(args.out, "xgb.pkl"))

if __name__ == "__main__":
    main()
