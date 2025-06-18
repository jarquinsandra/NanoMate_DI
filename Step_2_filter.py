import argparse
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from pycombat.pycombat import Combat
from statsmodels.nonparametric.smoothers_lowess import lowess
import time

# CLI
parser = argparse.ArgumentParser(description="Filter, normalize, impute, and visualize peak table.")
parser.add_argument("--input", required=True)
parser.add_argument("--tic_statistic", choices=["sum", "mean"], default="sum",
                    help="Statistic to use for TIC normalisation: sum or mean (default: sum)")
parser.add_argument("--batch_column", help="Metadata column containing batch labels for ComBat correction")
parser.add_argument("--metadata")
parser.add_argument("--group_column")
parser.add_argument("--mahal_threshold", type=float)
parser.add_argument("--neighbors", type=int, default=5)
parser.add_argument("--impute_method", choices=["knn", "rf"], default="knn")
parser.add_argument("--zero_threshold", type=float, default=0.8)
parser.add_argument("--log", choices=["log2", "log10", "ln"])
parser.add_argument("--outdir", default=".")
parser.add_argument("--plot_format", choices=["png", "pdf"], default="png")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--normalisation", choices=["none", "PQN", "TIC", "median"], default="PQN")
parser.add_argument("--qc_rlsc", action="store_true", help="Apply QC-RLSC drift correction")
parser.add_argument("--qc_label", default="QC", help="Label to identify QC samples in metadata")
parser.add_argument("--tic_stat", choices=["sum", "mean"], default="sum", help="Statistic for TIC normalization")
parser.add_argument("--run_order_column", help="Column in metadata with injection/run order")
parser.add_argument("--rlsc_frac", type=float, default=0.3,
                    help="LOESS smoothing fraction for QC-RLSC (default: 0.3)")
parser.add_argument("--skip-plots", action="store_true", help="Skip all plots")
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

def log_timed_step(label, start_time):
    elapsed = time.time() - start_time
    log(f"⏱️ {label} completed in {elapsed:.2f} seconds")


log(f"📥 Arguments parsed: {args}")
start_total = time.time()
print("✅ Step 1: Getting basename...")
basename = os.path.splitext(os.path.basename(args.input))[0]
print(f"✅ basename: {basename}")

print("✅ Step 2: Creating output dir...")
os.makedirs(args.outdir, exist_ok=True)
print(f"✅ Output directory created or exists: {args.outdir}")

print("✅ Step 3: Preparing log file path...")
log_path = os.path.join(args.outdir, f"{basename}_processing_log.txt")
print(f"✅ log_path: {log_path}")

print("✅ Step 4: Opening log file...")
log_file = open(log_path, "w", encoding="utf-8")
print("✅ Log file opened successfully")


def save_tic_plot(df, label, suffix):
    if metadata is not None:
        tic = df.sum(axis=1)
        grouped = pd.DataFrame({args.group_column: metadata[args.group_column], "TIC": tic})
        plt.figure(figsize=(6, 4))
        ax = sns.boxplot(
            data=grouped,
            x=args.group_column,
            y="TIC",
            hue=None,
            palette="Set2",
            legend=False
        )
        plt.title(f"TIC by Group - {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"{basename}_{suffix}.{args.plot_format}"))
        plt.close()
        group = grouped.groupby(args.group_column)
        stats = group["TIC"].agg(['mean', 'median', 'std', 'count'])
        stats['RSD'] = 100 * stats['std'] / (stats['mean'] + EPSILON)
        stats.to_csv(os.path.join(args.outdir, f"{basename}_{suffix}_TIC_stats.csv"))
        log(f"📦 Saved TIC plot and stats: {suffix}")



def save_median_comparison(before_df, after_df):
    plt.figure(figsize=(6, 4))
    plt.plot(before_df.median(axis=1), label="Before Normalisation")
    plt.plot(after_df.median(axis=1), label="After Normalisation")
    plt.title("Median Intensities per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Median Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{basename}_medians_before_after.{args.plot_format}"))
    plt.close()
    log("📈 Saved median comparison plot")

def post_imputation_qc(matrix):
    unique_counts = matrix.nunique()
    low_unique = (unique_counts <= 2).sum()
    low_var = (matrix.var() < 1e-6).sum()
    path = os.path.join(args.outdir, f"{basename}_imputation_qc.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Features with <=2 unique values: {low_unique}\n")
        f.write(f"Features with variance < 1e-6: {low_var}\n")
    log(f"🔍 Saved imputation QC: {path}")

def apply_qc_rlsc(df, metadata, qc_label="QC", order_col="Order"):
    if metadata is None or order_col not in metadata.columns:
        raise ValueError("Metadata with run order column is required for QC-RLSC.")
    run_order = metadata[order_col]
    qc_mask = metadata[args.group_column] == qc_label
    corrected_df = df.copy()
    skipped_features = []

    for feature in df.columns:
        y_qc = df.loc[qc_mask, feature]
        x_qc = run_order.loc[qc_mask]
        if y_qc.isnull().all() or y_qc.nunique() <= 1:
            skipped_features.append(feature)
            continue
        try:
            loess_fit = lowess(y_qc, x_qc, frac=args.rlsc_frac, return_sorted=False, it=3)
            median_qc = np.nanmedian(y_qc)
            full_loess = lowess(y_qc, x_qc, xvals=run_order, frac=args.rlsc_frac, it=3)
            correction_factor = np.clip(full_loess, a_min=EPSILON, a_max=None)
            corrected_values = df[feature] / correction_factor * median_qc
            corrected_df[feature] = corrected_values
        except Exception as e:
            log(f"⚠️ Skipping feature '{feature}' due to RLSC error: {e}")
            skipped_features.append(feature)

    if skipped_features:
        log(f"⚠️ Skipped {len(skipped_features)} features during RLSC due to invalid input or fitting errors.")
        skipped_path = os.path.join(args.outdir, f"{basename}_rlsc_skipped_features.csv")
        pd.Series(skipped_features).to_csv(skipped_path, index=False, header=["Skipped_Features"])
        log(f"📝 Skipped features saved to: {skipped_path}")

    return corrected_df
def plot_pca_by_batch(matrix, metadata, batch_col, label, suffix):
    if batch_col not in metadata.columns:
        log(f"⚠️ Skipping PCA: batch column '{batch_col}' not found.")
        return

    batch = metadata.loc[matrix.index, batch_col]
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(StandardScaler().fit_transform(matrix))
    pc_df = pd.DataFrame(pcs, index=matrix.index, columns=["PC1", "PC2"])
    pc_df["Batch"] = batch.values

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=pc_df, x="PC1", y="PC2", hue="Batch",
        palette="Set2", s=60, edgecolor="black"
    )
    plt.title(f"PCA by {batch_col} - {label}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{basename}_{suffix}_pca_by_{batch_col}.{args.plot_format}"))
    plt.close()
    log(f"📊 Saved PCA plot: {suffix}_pca_by_{batch_col}")

from sklearn.preprocessing import LabelEncoder

def apply_combat(df, metadata, batch_col):
    if batch_col not in metadata.columns:
        raise ValueError(f"Batch column '{batch_col}' not found in metadata.")

    # Extract and flatten batch labels
    batch = metadata.loc[df.index, batch_col].astype(str).values
    if isinstance(batch, pd.Series):
        batch = batch.to_numpy()
    if batch.ndim != 1:
        batch = batch.ravel()

    # Ensure data is float and a NumPy array
    data = df.astype(np.float32).to_numpy()

    print("🧪 data.shape:", data.shape)
    print("🧪 batch shape:", batch.shape)
    print("🧪 unique batches:", pd.Series(batch).unique())

    try:
        corrected = Combat().fit_transform(data, batch)
        return pd.DataFrame(corrected, index=df.index, columns=df.columns)
    except Exception as e:
        raise RuntimeError(f"ComBat failed: {e}")

# Start processing


print("📂 Attempting to load peak table...")
start = time.time()

log(f"📂 Loading input file: {args.input}")
df = pd.read_csv(args.input, index_col=0, low_memory=False)
log(f"✅ CSV loaded: shape {df.shape}")
   
if args.metadata and args.group_column:
    print("📂 Attempting to load metadata...")
    metadata = pd.read_csv(args.metadata, index_col=0)
    print(f"✅ Metadata loaded: {metadata.shape}")

    # Clean sample names
    df.index = (
        df.index.astype(str)
        .str.replace(".mzML", "", case=False, regex=False)
        .str.strip()
        .str.lower()
    )
    metadata.index = (
        metadata.index.astype(str)
        .str.replace(".mzML", "", case=False, regex=False)
        .str.strip()
        .str.lower()
    )

    # Align
    common_samples = df.index.intersection(metadata.index)
    df = df.loc[common_samples]
    metadata = metadata.loc[common_samples]

    log(f"🧩 Matched samples between peak table and metadata: {len(common_samples)}")
    if len(common_samples) == 0:
        log("❌ ERROR: No matching sample names between metadata and input table.")
        log_file.close()
        exit(1)
else:
    metadata = None


    # Clean and match sample names between metadata and peak table
    metadata.index = (
        metadata.index.astype(str)
        .str.replace(".mzML", "", case=False, regex=False)
        .str.strip()
        .str.lower()
    )

    df.index = (
        df.index.astype(str)
        .str.replace(".mzML", "", case=False, regex=False)
        .str.strip()
        .str.lower()
    )

    common_samples = df.index.intersection(metadata.index)
    df = df.loc[common_samples]
    metadata = metadata.loc[common_samples]

    log(f"🧩 Matched samples between peak table and metadata: {len(common_samples)}")
    if len(common_samples) == 0:
        log("❌ ERROR: No matching sample names between metadata and input table.")
        log_file.close()
        exit(1)


log(f"📊 Input peak table shape: {df.shape}")
if metadata is not None:
    log(f"📑 Metadata shape: {metadata.shape}, columns: {metadata.columns.tolist()}")
log_timed_step("Data loading", start)

if not args.skip_plots:
    save_tic_plot(df, "Raw", "tic_raw")

# Zero filtering
start = time.time()
zero_fraction = (df.values == 0).sum(axis=0) / df.shape[0]
zero_fraction = pd.Series(zero_fraction, index=df.columns)
filtered_df = df.loc[:, zero_fraction <= args.zero_threshold]
log(f"🧮 Features before filtering: {df.shape[1]}")
log(f"🧮 Features after filtering: {filtered_df.shape[1]}")
log(f"🧪 Any features 100% zero? {(zero_fraction == 1).sum()}")
log(f"📉 Top 10 most zero-sparse features:\n{zero_fraction.sort_values(ascending=False).head(10)}")
log(f"🧹 Zero filtering complete. Retained {filtered_df.shape[1]} features from {df.shape[1]}.")
log_timed_step("Zero filtering", start)

# Normalization
start = time.time()
log(f"🔄 Starting normalization with method: {args.normalisation}")
if args.normalisation == "none":
    norm_df = filtered_df.copy()
elif args.normalisation == "PQN":
    pqn_df = filtered_df.replace(0, np.nan)
    reference = pqn_df.median()
    quotients = pqn_df.divide(reference)
    factors = quotients.median(axis=1)
    norm_df = pqn_df.divide(factors, axis=0)
    log(f"🔍 PQN result shape: {norm_df.shape}")
    log(f"🔍 All NaN after PQN? {norm_df.isna().all().all()}")
    log(f"🔍 Any entirely zero rows? {(norm_df.fillna(0) == 0).all(axis=1).sum()}")
    if not args.skip_plots:
        save_median_comparison(filtered_df, norm_df)
    if not args.skip_plots:
        save_tic_plot(norm_df, "After PQN", "tic_pqn")
elif args.normalisation == "TIC":
    if args.tic_stat == "sum":
        totals = filtered_df.sum(axis=1)
    else:
        totals = filtered_df.mean(axis=1)
    norm_df = filtered_df.div(totals, axis=0) * totals.median()
    save_median_comparison(filtered_df, norm_df)
    save_tic_plot(norm_df, "After TIC", "tic_tic")
elif args.normalisation == "median":
    medians = filtered_df.median(axis=1)
    norm_df = filtered_df.div(medians, axis=0) * medians.median()
    if not args.skip_plots:
        save_median_comparison(filtered_df, norm_df)
    if not args.skip_plots:
        save_tic_plot(norm_df, "After Median", "tic_median")
log(f"✅ Normalization ({args.normalisation}) complete.")
log_timed_step("Normalization", start)

# QC-RLSC
if args.qc_rlsc:
    start = time.time()
    log(f"🔧 Starting QC-RLSC with QC label '{args.qc_label}' and run order column '{args.run_order_column}'")
    norm_df = apply_qc_rlsc(norm_df, metadata, qc_label=args.qc_label, order_col=args.run_order_column)
    if not args.skip_plots:
        save_tic_plot(norm_df, "After QC-RLSC", "tic_rlsc")
    log_timed_step("QC-RLSC", start)
log(f"🔍 After QC-RLSC: {norm_df.shape}")
log(f"🔍 Any NaNs? {norm_df.isna().any().any()}")
log(f"🔍 All NaN columns? {norm_df.isna().all().sum()}")
log(f"🔍 All-zero columns? {(norm_df.fillna(0) == 0).all().sum()}")

# Mahalanobis
if args.mahal_threshold:
    start = time.time()
    log(f"📉 Starting Mahalanobis filtering with threshold {args.mahal_threshold}")
    scaled = StandardScaler().fit_transform(norm_df.fillna(0))
    pca = PCA(n_components=2).fit_transform(scaled)
    center = pca.mean(axis=0)
    cov = np.cov(pca, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    mahal_d = [distance.mahalanobis(x, center, inv_cov) for x in pca]
    keep = np.array(mahal_d) < args.mahal_threshold
    removed = norm_df.index[~keep].tolist()
    norm_df = norm_df.loc[keep]
    if metadata is not None:
        metadata = metadata.loc[norm_df.index]
    log(f"✅ Mahalanobis filtering complete. Removed {len(removed)} samples: {removed}")
    if not args.skip_plots:
        save_tic_plot(norm_df, "After Mahalanobis", "tic_mahal")
    log_timed_step("Mahalanobis filtering", start)


#####Combat correction

log(f"🔎 Variance summary before ComBat:\n{norm_df.var().describe()}")

if args.batch_column is None:
    log("⚠️ No batch column specified (--batch_column). Skipping ComBat correction.")
else:
    log(f"🔎 Batch sizes:\n{metadata.loc[norm_df.index, args.batch_column].value_counts()}")
    log("🧹 Removing features with no variance within any batch...")

    batch_stds = norm_df.groupby(metadata.loc[norm_df.index, args.batch_column]).std()
    features_with_zero_std = batch_stds.columns[(batch_stds == 0).any(axis=0)]
    if not features_with_zero_std.empty:
        log(f"🧹 Dropping {len(features_with_zero_std)} features with zero variance within at least one batch.")
        norm_df = norm_df.drop(columns=features_with_zero_std)

    if norm_df.isna().any().any():
        log("🧪 Detected NaNs before ComBat — applying temporary mean imputation")
        valid_cols = norm_df.columns[norm_df.notna().any()]
        dropped_cols = norm_df.columns.difference(valid_cols)
        log(f"🧹 Dropping {len(dropped_cols)} columns with all-NaN values before imputation.")
        norm_df_valid = norm_df[valid_cols]
        imputer = SimpleImputer(strategy="mean")
        norm_df_imputed = imputer.fit_transform(norm_df_valid)
        norm_df = pd.DataFrame(norm_df_imputed, columns=valid_cols, index=norm_df.index)
        log(f"✅ Temporary imputation complete. Columns retained: {len(valid_cols)}")

    log(f"🔢 Shape before ComBat: {norm_df.shape}")

    if norm_df.shape[1] < 10:
        log(f"⚠️ Only {norm_df.shape[1]} features remaining before ComBat. Skipping batch correction.")
    else:
        try:
            if not args.skip_plots:
                plot_pca_by_batch(norm_df, metadata, args.batch_column, label="Before ComBat", suffix="before_combat")

            norm_df = apply_combat(norm_df.copy(), metadata, args.batch_column)

            if not args.skip_plots:
                plot_pca_by_batch(norm_df, metadata, args.batch_column, label="After ComBat", suffix="after_combat")

            log(f"✅ ComBat correction complete. Shape: {norm_df.shape}")

        except Exception as e:
            log(f"❌ Error during ComBat correction: {e}")
            log_file.close()
            exit(1)
# Imputation
start = time.time()
# Imputation
start = time.time()
log("⏳ Starting imputation...")
log(f"🔢 Imputation method: {args.impute_method}")
# Remove all-zero columns
zero_cols = (norm_df == 0).all(axis=0)
num_zero_cols = zero_cols.sum()
if num_zero_cols > 0:
    print(f"🧹 Removing {num_zero_cols} all-zero columns...")
    norm_df = norm_df.loc[:, ~zero_cols]

log(f"📊 norm_df shape before imputation: {norm_df.shape}")
log(f"🧪 Any NaNs? {norm_df.isna().any().any()}")
log(f"🧪 All zero? {(norm_df == 0).all().all()}")

# Drop all-NaN or all-zero rows or columns
norm_df = norm_df.dropna(axis=0, how="all")
norm_df = norm_df.loc[(norm_df != 0).any(axis=1)]
norm_df = norm_df.dropna(axis=1, how="all")

# Log again after cleaning
log(f"🧼 Cleaned norm_df shape: {norm_df.shape}")
if norm_df.empty:
    log("❌ ERROR: No usable data remains after cleaning. Aborting.")
    log_file.close()
    exit(1)

# Proceed to scale/impute

if args.impute_method == "knn":
    log("🔄 Imputing with kNN...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(norm_df)
    imputer = KNNImputer(n_neighbors=args.neighbors)
    imputed = imputer.fit_transform(scaled)
    df_imputed = pd.DataFrame(scaler.inverse_transform(imputed), index=norm_df.index, columns=norm_df.columns)
elif args.impute_method == "rf":
    log("⚠️ WARNING: Random Forest imputation may take a long time.")
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
                               max_iter=10, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(norm_df), index=norm_df.index, columns=norm_df.columns)
log_timed_step("Imputation", start)

# Log transform
# Safe Log transform
if args.log:
    start = time.time()
    log(f"🧮 Log transform ({args.log}) applied.")

    # Clip to avoid log of zero or negative numbers
    df_imputed = df_imputed.clip(lower=1e-9)

    if args.log == "log2":
        df_imputed = np.log2(df_imputed)
    elif args.log == "log10":
        df_imputed = np.log10(df_imputed)
    elif args.log == "ln":
        df_imputed = np.log(df_imputed)

    log_timed_step("Log transformation", start)

# Save final matrix
start = time.time()
final_out = os.path.join(args.outdir, f"{basename}_filtered_imputed.csv")
log(f"💾 Writing final imputed matrix to: {final_out}")
df_imputed.to_csv(final_out)
if not args.skip_plots:
    save_tic_plot(df_imputed, "After Imputation", "tic_final")
post_imputation_qc(df_imputed)

log_timed_step("Final export and post-QC", start)

log_timed_step("Total pipeline", start_total)
log("✅ Processing complete.")
log_file.close()