#!/usr/bin/env python3
"""
Heavy-Metal-Prediction-in-Groundwater — Synthetic Demo

Generates a synthetic groundwater dataset (>100 rows), trains a
RandomForestRegressor to predict Pb_ppb or As_ppb, and saves artifacts.
"""
from __future__ import annotations
import argparse, os, warnings
import numpy as np, pandas as pd
from numpy.random import default_rng
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

def _bounded_normal(mean, sd, low, high, n, rng):
    vals = rng.normal(mean, sd, n)
    return np.clip(vals, low, high)

def generate_synthetic_groundwater(n=1000, seed=42):
    rng = default_rng(seed)
    pH = _bounded_normal(7.1, 0.6, 5.5, 8.8, n, rng)
    EC_uScm = rng.gamma(3.0, 250.0, n); EC_uScm = np.clip(EC_uScm, 100, 4000)
    TDS_mgL = np.clip(EC_uScm * rng.normal(0.65, 0.08, n), 80, 3000)
    Hardness_mgL = np.clip(rng.normal(200, 80, n), 40, 600)
    Nitrate_mgL = np.clip(rng.gamma(2.0, 6.0, n), 0, 80)
    Sulfate_mgL = np.clip(rng.gamma(2.0, 30.0, n), 0, 500)
    Chloride_mgL = np.clip(rng.gamma(2.5, 35.0, n), 5, 700)
    DO_mgL = np.clip(rng.normal(5.5, 1.8, n), 0.5, 12)
    Depth_m = np.clip(rng.normal(50, 25, n), 2, 180)
    Distance_to_industry_km = np.clip(rng.exponential(5.0, n), 0.0, 30.0)
    Temperature_C = np.clip(rng.normal(27.0, 3.0, n), 15, 40)
    Rainfall_mm = np.clip(rng.gamma(2.0, 40.0, n), 0, 400)
    Landuse = rng.choice(np.array(["agric","urban","industrial"]), size=n, p=[0.45,0.35,0.20])
    lu_w = np.vectorize({"agric":1.0,"urban":1.4,"industrial":2.0}.get)(Landuse)
    inv_depth = 1.0/(Depth_m+5.0); inv_dist = 1.0/(Distance_to_industry_km+0.5)
    Pb_ppb = (4.0 + 8.0*np.maximum(0,7.0-pH) + 0.006*EC_uScm + 0.004*TDS_mgL +
              20.0*inv_depth + 25.0*inv_dist + 0.6*Nitrate_mgL + 0.03*Sulfate_mgL +
              5.0*(lu_w-1.0) + rng.normal(0,6.0,n))
    Pb_ppb = np.clip(Pb_ppb, 0, None)
    As_ppb = (3.0 + 0.004*EC_uScm + 0.002*TDS_mgL + 18.0*inv_depth +
              0.4*Sulfate_mgL + 0.3*Chloride_mgL + 0.4*Nitrate_mgL +
              4.0*(lu_w-1.0) + rng.normal(0,5.0,n))
    As_ppb = np.clip(As_ppb, 0, None)
    df = pd.DataFrame({\"pH\":pH,\"EC_uScm\":EC_uScm,\"TDS_mgL\":TDS_mgL,\"Hardness_mgL\":Hardness_mgL,
                       \"Nitrate_mgL\":Nitrate_mgL,\"Sulfate_mgL\":Sulfate_mgL,\"Chloride_mgL\":Chloride_mgL,
                       \"DO_mgL\":DO_mgL,\"Depth_m\":Depth_m,\"Distance_to_industry_km\":Distance_to_industry_km,
                       \"Temperature_C\":Temperature_C,\"Rainfall_mm\":Rainfall_mm,\"Landuse\":Landuse,
                       \"Pb_ppb\":Pb_ppb,\"As_ppb\":As_ppb})
    df[\"Pb_exceeds_WHO_10ppb\"] = (df[\"Pb_ppb\"] > 10).astype(int)
    df[\"As_exceeds_WHO_10ppb\"] = (df[\"As_ppb\"] > 10).astype(int)
    return df

def build_and_train(df, target=\"Pb_ppb\", seed=42):
    feature_cols = [\"pH\",\"EC_uScm\",\"TDS_mgL\",\"Hardness_mgL\",\"Nitrate_mgL\",\"Sulfate_mgL\",
                    \"Chloride_mgL\",\"DO_mgL\",\"Depth_m\",\"Distance_to_industry_km\",
                    \"Temperature_C\",\"Rainfall_mm\",\"Landuse\"]
    X = df[feature_cols]; y = df[target]
    num_cols = [c for c in feature_cols if c != \"Landuse\"]
    pre = ColumnTransformer([(\"num\", StandardScaler(), num_cols),
                             (\"cat\", OneHotEncoder(handle_unknown=\"ignore\"), [\"Landuse\"])])    
    model = Pipeline([(\"pre\", pre),
                      (\"rf\", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=1337))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    model.fit(Xtr, ytr)
    cv = cross_val_score(model, Xtr, ytr, cv=5, scoring=\"r2\", n_jobs=-1)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred); rmse = mean_squared_error(yte, pred, squared=False); mae = mean_absolute_error(yte, pred)
    return model, {\"cv_mean_r2\":float(cv.mean()), \"cv_std\":float(cv.std()), \"r2\":float(r2), \"rmse\":float(rmse), \"mae\":float(mae)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--n\", type=int, default=1000)
    ap.add_argument(\"--seed\", type=int, default=42)
    ap.add_argument(\"--target\", type=str, default=\"Pb_ppb\", choices=[\"Pb_ppb\",\"As_ppb\"])    
    args = ap.parse_args()
    if args.n < 101: raise SystemExit(\"Please use --n >= 101 (requirement: >100 points)\")
    df = generate_synthetic_groundwater(n=args.n, seed=args.seed)
    os.makedirs(\"artifacts\", exist_ok=True)
    df.to_csv(\"artifacts/synthetic_groundwater.csv\", index=False)
    model, metrics = build_and_train(df, target=args.target, seed=args.seed)
    import joblib; joblib.dump(model, f\"artifacts/model_{args.target}.joblib\")
    print(\"Rows:\", len(df), \" Target:\", args.target, \" Metrics:\", metrics)

if __name__ == \"__main__\":
    main()
