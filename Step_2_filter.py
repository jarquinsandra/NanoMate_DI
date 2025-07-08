# Full-featured Step_2_filter.py with fast performance and optional log + no impute
# Based on user's original version

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from pycombat.pycombat import Combat
from statsmodels.nonparametric.smoothers_lowess import lowess
import time

# CLI
parser = argparse.ArgumentParser(description="Filter, normalize, correct, and optionally log-transform a peak table.")
parser.add_argument("--input", required=True)
parser.add_argument("--metadata")
parser.add_argument("--group_column")
parser.add_argument("--batch_column")
parser.add_argument("--run_order_column")
parser.add_argument("--qc_label", default="QC")
parser.add_argument("--qc_rlsc", action="store_true")
parser.add_argument("--rlsc_frac", type=float, default=0.3)
parser.add_argument("--normalisation", choices=["none", "PQN", "TIC", "median"], default="PQN")
parser.add_argument("--tic_statistic", choices=["sum", "mean"], default="sum")
parser.add_argument("--zero_threshold", type=float, default=0.8)
parser.add_argument("--mahal_threshold", type=float)
parser.add_argument("--no_impute", action="store_true")
parser.add_argument("--impute_method", choices=["knn", "rf"], default="knn")
parser.add_argument("--neighbors", type=int, default=5)
parser.add_argument("--log", choices=["log2", "log10", "ln"])
parser.add_argument("--clip_negatives", action="store_true")
parser.add_argument("--outdir", default=".")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

EPSILON = 1e-9
basename = os.path.splitext(os.path.basename(args.input))[0]
os.makedirs(args.outdir, exist_ok=True)
log_path = os.path.join(args.outdir, f"{basename}_processing_log.txt")
log_file = open(log_path, "w", encoding="utf-8")

def log(msg):
    if args.verbose:
        print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

def log_timed_step(label, start):
    log(f"⏱️ {label} completed in {time.time() - start:.2f} sec")

start_total = time.time()
log(f"📥 Loading input: {args.input}")
df = pd.read_csv(args.input, index_col=0, low_memory=False)
metadata = pd.read_csv(args.metadata, index_col=0) if args.metadata else None
if metadata is not None:
    df.index = df.index.astype(str).str.replace(".mzML", "", case=False).str.strip().str.lower()
    metadata.index = metadata.index.astype(str).str.replace(".mzML", "", case=False).str.strip().str.lower()
    df = df.loc[df.index.intersection(metadata.index)]
    metadata = metadata.loc[df.index]
    log(f"🧩 Matched {len(df)} samples")

log(f"🔢 Initial shape: {df.shape}")

# Zero filtering
start = time.time()
zero_fraction = (df == 0).sum(axis=0) / df.shape[0]
df = df.loc[:, zero_fraction <= args.zero_threshold]
log(f"🧹 Zero filtering: retained {df.shape[1]} features")
log_timed_step("Zero filtering", start)

# Normalization
start = time.time()
if args.normalisation == "none":
    norm_df = df.copy()
elif args.normalisation == "PQN":
    pqn_df = df.replace(0, np.nan)
    ref = pqn_df.median()
    quotients = pqn_df.divide(ref)
    factors = quotients.median(axis=1)
    norm_df = pqn_df.divide(factors, axis=0)
elif args.normalisation == "TIC":
    totals = df.sum(axis=1) if args.tic_statistic == "sum" else df.mean(axis=1)
    norm_df = df.div(totals, axis=0) * totals.median()
elif args.normalisation == "median":
    medians = df.median(axis=1)
    norm_df = df.div(medians, axis=0) * medians.median()
log_timed_step("Normalization", start)

# QC-RLSC (sequential version)
if args.qc_rlsc and metadata is not None and args.run_order_column:
    start = time.time()
    log("🔧 Running QC-RLSC (sequential)")
    run_order = metadata[args.run_order_column]
    qc_mask = metadata[args.group_column] == args.qc_label
    for feature in norm_df.columns:
        y_qc = norm_df.loc[qc_mask, feature]
        x_qc = run_order.loc[qc_mask]
        if y_qc.isnull().all() or y_qc.nunique() <= 1:
            continue
        try:
            smoothed = lowess(y_qc, x_qc, xvals=run_order, frac=args.rlsc_frac, it=3)
            norm_df[feature] = norm_df[feature] / np.clip(smoothed, a_min=EPSILON, a_max=None) * np.nanmedian(y_qc)
        except Exception as e:
            log(f"⚠️ RLSC failed for {feature}: {e}")
    log_timed_step("QC-RLSC", start)

# Mahalanobis outlier filtering
if args.mahal_threshold:
    start = time.time()
    scaled = StandardScaler().fit_transform(norm_df.fillna(0))
    pca = PCA(n_components=2).fit_transform(scaled)
    center = pca.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(pca, rowvar=False))
    mahal_d = [distance.mahalanobis(x, center, inv_cov) for x in pca]
    keep = np.array(mahal_d) < args.mahal_threshold
    norm_df = norm_df.loc[keep]
    metadata = metadata.loc[norm_df.index] if metadata is not None else None
    log(f"📉 Mahalanobis removed {len(mahal_d) - keep.sum()} samples")
    log_timed_step("Mahalanobis filtering", start)

# ComBat batch correction
if args.batch_column and metadata is not None:
    start = time.time()
    batch = metadata.loc[norm_df.index, args.batch_column].astype(str).values
    if norm_df.isna().any().any():
        imputer = SimpleImputer(strategy="mean")
        norm_df = pd.DataFrame(imputer.fit_transform(norm_df), index=norm_df.index, columns=norm_df.columns)
    norm_df = pd.DataFrame(Combat().fit_transform(norm_df.values, batch), index=norm_df.index, columns=norm_df.columns)
    log_timed_step("ComBat correction", start)

# Log transform (optional)
if args.log:
    start = time.time()
    norm_df = norm_df.clip(lower=EPSILON)
    if args.log == "log2":
        norm_df = np.log2(norm_df)
    elif args.log == "log10":
        norm_df = np.log10(norm_df)
    elif args.log == "ln":
        norm_df = np.log(norm_df)
    log_timed_step("Log transformation", start)

# Clip negatives if requested
if args.clip_negatives:
    norm_df = norm_df.clip(lower=0)

# Imputation
if args.no_impute:
    log("🚫 Skipping imputation (using --no_impute)")
else:
    start = time.time()
    log(f"🔢 Starting imputation using method: {args.impute_method}")
    if args.impute_method == "knn":
        scaler = StandardScaler()
        scaled = scaler.fit_transform(norm_df)
        imputer = KNNImputer(n_neighbors=args.neighbors)
        imputed = imputer.fit_transform(scaled)
        norm_df = pd.DataFrame(scaler.inverse_transform(imputed), index=norm_df.index, columns=norm_df.columns)
    elif args.impute_method == "rf":
        imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
                                   max_iter=10, random_state=42)
        norm_df = pd.DataFrame(imputer.fit_transform(norm_df), index=norm_df.index, columns=norm_df.columns)
    log_timed_step("Imputation", start)

# Export
output_path = os.path.join(args.outdir, f"{basename}_filtered.csv")
norm_df.to_csv(output_path)
log(f"✅ Final matrix saved: {output_path}")
log_timed_step("Total pipeline", start_total)
log_file.close()
