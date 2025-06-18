import argparse 
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from pathlib import Path
import hashlib
from sklearn.manifold import TSNE
from umap import UMAP

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="QC evaluation with PCA-based outlier filtering.")
parser.add_argument("--umap", action="store_true", help="Generate UMAP plot")
parser.add_argument("--tsne", action="store_true", help="Generate t-SNE plot")
parser.add_argument("--plot_format", choices=["png", "pdf"], default="png", help="Output image format")
parser.add_argument("--input", required=True, help="Aligned peak table (samples as rows)")
parser.add_argument("--metadata", required=True, help="Metadata CSV")
parser.add_argument("--sample_type_column", required=True, help="Metadata column labeling sample type")
parser.add_argument("--qc_label", required=True, help="Value used to identify QC samples")
parser.add_argument("--qa_label", help="Value used to identify QA samples (optional)")
parser.add_argument("--plate_column", required=True, help="Metadata column for coloring samples")
parser.add_argument("--outdir", default="qc_evaluation_results", help="Directory to save output")
parser.add_argument("--mahal_threshold", type=float, default=3.5, help="Mahalanobis distance threshold")
parser.add_argument("--export_cleaned_data", action="store_true", help="Save cleaned peak table and metadata")
parser.add_argument("--pca_batch_size", type=int, default=100, help="Batch size for Incremental PCA")
parser.add_argument("--subset_sample_count", type=int, help="Randomly sample this many non-QC samples")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--plot_qc_drift", action="store_true", help="Plot top features in QCs across injection order")
parser.add_argument("--run_order_column", default="RunOrder", help="Column in metadata indicating injection/run order")
parser.add_argument("--no_filtering", action="store_true", help="Skip Mahalanobis outlier filtering")
parser.add_argument("--log_transform", action="store_true", help="Apply log2(x+1) transformation to input data")
parser.add_argument("--plot_all", action="store_true", help="Plot all samples even if >300")
parser.add_argument("--tsne_perplexity", type=float, default=30, help="Perplexity for t-SNE")
parser.add_argument("--umap_neighbors", type=int, default=15, help="UMAP number of neighbors")
parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP minimum distance")
parser.add_argument("--exclude_sample_types", nargs="*", help="List of sample types to exclude")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
args = parser.parse_args()

# ---------- Setup ----------
EPSILON = 1e-9
np.random.seed(args.random_seed)
os.makedirs(args.outdir, exist_ok=True)
basename = os.path.splitext(os.path.basename(args.input))[0]
CACHE_DIR = Path(args.outdir) / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

def log(msg):
    if args.verbose:
        print(msg)

log("🔄 Hashing input file...")
def hash_file(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
input_hash = hash_file(args.input)
cache_file = CACHE_DIR / f"scaled_pca_{input_hash}.pkl"

use_cache = False
if cache_file.exists():
    try:
        log("⚡ Loading cached PCA data...")
        df, metadata, scaled, pca_data, explained = load(cache_file)
        use_cache = True
    except Exception as e:
        log(f"⚠️ Failed to load cache ({e}). Recomputing PCA...")

if not use_cache:
    metadata = pd.read_csv(args.metadata)
    metadata.iloc[:, 0] = metadata.iloc[:, 0].str.replace(" ", "", regex=False)
    metadata.set_index(metadata.columns[0], inplace=True)
    metadata.index = metadata.index.str.replace(".mzML", "", regex=False).str.strip().str.lower()
    metadata[args.sample_type_column] = metadata[args.sample_type_column].astype(str).str.strip().str.upper()
    args.qc_label = args.qc_label.strip().upper()
    args.qa_label = args.qa_label.strip().upper() if args.qa_label else None

    if args.exclude_sample_types:
        exclude = [x.strip().upper() for x in args.exclude_sample_types]
        metadata = metadata[~metadata[args.sample_type_column].isin(exclude)]
        log(f"🚫 Excluded sample types: {exclude}")

    qc_mask = metadata[args.sample_type_column] == args.qc_label
    qc_samples = metadata[qc_mask].index.tolist()
    non_qc_samples = metadata[~qc_mask].index.tolist()
    if args.subset_sample_count:
        random_subset = np.random.choice(non_qc_samples, size=min(args.subset_sample_count, len(non_qc_samples)), replace=False)
        selected_samples = set(qc_samples + list(random_subset))
    else:
        selected_samples = set(metadata.index)

    df_chunks = []
    for chunk in pd.read_csv(args.input, index_col=0, chunksize=1000):
        chunk.index = chunk.index.str.replace(".mzML", "", regex=False).str.strip().str.lower()
        filtered = chunk.loc[chunk.index.intersection(selected_samples)]
        if not filtered.empty:
            df_chunks.append(filtered)
    df_full = pd.concat(df_chunks)
    metadata = metadata.loc[df_full.index]
    df = df_full

    numeric_data = df.select_dtypes(include=[np.number])
    if args.log_transform:
        log_df = np.log2(numeric_data.clip(lower=0) + 1).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    else:
        log_df = numeric_data.astype(np.float32)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_df)
    ipca = IncrementalPCA(n_components=2, batch_size=args.pca_batch_size)
    pca_data = ipca.fit_transform(scaled)
    explained = ipca.explained_variance_ratio_ * 100
    dump((df, metadata, scaled, pca_data, explained), cache_file)

# Create PCA DataFrame
pca_df = pd.DataFrame(pca_data, index=df.index, columns=["PC1", "PC2"])
pca_df["Plate"] = metadata[args.plate_column]
pca_df["SampleType"] = metadata[args.sample_type_column]
palette = sns.color_palette("Set2", len(pca_df["Plate"].unique()))
plate_colors = dict(zip(pca_df["Plate"].unique(), palette))

def scatter_with_qc_overlay(df_proj, x, y, filename, title):
    is_qc = df_proj["SampleType"] == args.qc_label
    non_qc = df_proj[~is_qc]
    qc = df_proj[is_qc]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=non_qc, x=x, y=y, hue="Plate", palette=plate_colors,
        s=60, edgecolor="black", marker="o"
    )
    sns.scatterplot(
        data=qc, x=x, y=y, color="black", marker="^", s=90,
        label="QC", edgecolor="black", zorder=10
    )
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, filename))
    plt.close()

log("📊 Generating PCA plot...")
scatter_with_qc_overlay(pca_df, "PC1", "PC2",
    f"{basename}_PCA_by_plate_qc_shape.{args.plot_format}",
    "PCA (QC shape, Plate color)")

if args.umap:
    reducer = UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist, random_state=args.random_seed)
    umap_data = reducer.fit_transform(scaled)
    umap_df = pd.DataFrame(umap_data, index=df.index, columns=["UMAP1", "UMAP2"])
    umap_df["Plate"] = metadata[args.plate_column]
    umap_df["SampleType"] = metadata[args.sample_type_column]
    scatter_with_qc_overlay(umap_df, "UMAP1", "UMAP2",
        f"{basename}_UMAP_by_plate_qc_shape.{args.plot_format}",
        "UMAP (QC shape, Plate color)")

if args.tsne:
    tsne = TSNE(n_components=2, random_state=args.random_seed, perplexity=args.tsne_perplexity)
    tsne_data = tsne.fit_transform(scaled)
    tsne_df = pd.DataFrame(tsne_data, index=df.index, columns=["tSNE1", "tSNE2"])
    tsne_df["Plate"] = metadata[args.plate_column]
    tsne_df["SampleType"] = metadata[args.sample_type_column]
    scatter_with_qc_overlay(tsne_df, "tSNE1", "tSNE2",
        f"{basename}_tSNE_by_plate_qc_shape.{args.plot_format}",
        "t-SNE (QC shape, Plate color)")

log("📐 Calculating Mahalanobis distances...")
center = np.mean(pca_data, axis=0)
cov = np.cov(pca_data, rowvar=False)
inv_covmat = np.linalg.pinv(cov)
diff = pca_data - center
if args.no_filtering:
    log("🚫 Skipping Mahalanobis filtering...")
    outlier_mask = np.array([False] * len(df))
else:
    mahal_d = np.sqrt(np.sum(diff @ inv_covmat * diff, axis=1))
    pca_df["Mahalanobis"] = mahal_d
    outlier_mask = mahal_d >= args.mahal_threshold

# Save Mahalanobis distances
pca_df[["Mahalanobis"]].to_csv(os.path.join(args.outdir, f"{basename}_Mahalanobis_distances.csv"))

# Save outlier list
with open(os.path.join(args.outdir, f"{basename}_outliers.txt"), "w") as f:
    for sample in sorted(df.index[outlier_mask]):
        f.write(sample + "\n")

# Plot PCA with outliers
log("📉 Plotting PCA with outliers highlighted...")
pca_df["Outlier"] = outlier_mask
plt.figure(figsize=(6, 5))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Outlier", palette={True: "red", False: "gray"}, s=60, edgecolor="black")
plt.xlabel(f"PC1 ({explained[0]:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]:.1f}%)")
plt.title("PCA with Mahalanobis Outliers Highlighted")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, f"{basename}_PCA_outliers_marked.{args.plot_format}"))
plt.close()

log("🧹 Filtering outliers...")
df_clean = df.loc[~outlier_mask]
metadata_clean = metadata.loc[df_clean.index]

# Export cleaned data
if args.export_cleaned_data:
    df_clean.to_csv(os.path.join(args.outdir, f"{basename}_cleaned.csv"))
    metadata_clean.to_csv(os.path.join(args.outdir, f"{basename}_metadata_cleaned.csv"))
    pca_df.loc[~outlier_mask, ["PC1", "PC2"]].to_csv(os.path.join(args.outdir, f"{basename}_PCA_filtered.csv"))

# QC RSD
log("📏 Computing RSDs for QC samples...")
qc_filtered = metadata_clean[metadata_clean[args.sample_type_column] == args.qc_label]
df_qc = df_clean.loc[qc_filtered.index]
if len(df_qc) >= 2:
    rsd_values = np.std(df_qc.values, axis=0) / (np.mean(df_qc.values, axis=0) + EPSILON) * 100
    rsd_series = pd.Series(rsd_values, index=df_qc.columns, name="RSD_%")
    rsd_series.reset_index().rename(columns={"index": "m/z"}).to_csv(
        os.path.join(args.outdir, f"{basename}_QC_feature_RSDs.csv"), index=False)
    plt.figure(figsize=(6, 4))
    sns.histplot(rsd_series, bins=50, kde=False)
    plt.title("RSD Distribution (QC Samples)")
    plt.xlabel("RSD (%)")
    plt.ylabel("Number of Features")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{basename}_QC_RSD_distribution.{args.plot_format}"))
    plt.close()

# Summary report
summary_path = os.path.join(args.outdir, f"{basename}_qc_summary.txt")
with open(summary_path, "w") as f:
    total = len(df)
    removed = outlier_mask.sum()
    remaining = len(df_clean)
    f.write(f"Summary Report:\n- Total samples: {total}\n- Outliers removed: {removed}\n- Remaining: {remaining}\n")
    rsd_path = os.path.join(args.outdir, f"{basename}_QC_feature_RSDs.csv")
    if os.path.exists(rsd_path):
        rsd_df = pd.read_csv(rsd_path)
        median_rsd = rsd_df["RSD_%"].median()
        pct_below_30 = (rsd_df["RSD_%"] < 30).mean() * 100
        f.write(f"- Median RSD (QC): {median_rsd:.2f}%\n")
        f.write(f"- Features with RSD < 30%: {pct_below_30:.1f}%\n")
    else:
        f.write("- RSD file not found.\n")

log(f"✅ Summary report written to: {summary_path}")

# Optional drift plot
def plot_qc_drift_lineplot(df_qc, metadata_qc, run_order_col, outdir, basename, top_n=5):
    import matplotlib.cm as cm
    from scipy.ndimage import uniform_filter1d

    # Ensure only QC samples
    df_qc = df_qc.loc[metadata_qc[metadata_qc[args.sample_type_column] == args.qc_label].index]
    metadata_qc = metadata_qc.loc[df_qc.index]

    # Sort by injection order
    metadata_qc_sorted = metadata_qc.sort_values(run_order_col)
    df_qc_sorted = df_qc.loc[metadata_qc_sorted.index].clip(lower=0)

    # Remove features with very low signal
    df_qc_sorted = df_qc_sorted.loc[:, df_qc_sorted.max() > 1]

    # Select top N most intense features
    top_features = df_qc_sorted.mean().sort_values(ascending=False).head(top_n).index

    plt.figure(figsize=(8, 5))
    colors = cm.get_cmap("tab10", top_n)

    for i, feature in enumerate(top_features):
        x = metadata_qc_sorted[run_order_col]
        y = df_qc_sorted[feature]
        y_smoothed = uniform_filter1d(y, size=3)  # simple smoothing
        sns.lineplot(x=x, y=y_smoothed, label=f"{feature}", color=colors(i), marker="o")

    plt.xlabel("Injection Order")
    plt.ylabel("Intensity")
    plt.title(f"Top {top_n} QC Features vs Injection Order")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(outdir, f"{basename}_qc_drift_top{top_n}.png")
    plt.savefig(path)
    plt.close()
    log(f"📈 QC drift plot saved: {path}")

if args.plot_qc_drift and args.run_order_column in metadata_clean.columns:
    log("📉 Plotting QC drift vs injection order...")
    plot_qc_drift_lineplot(df_qc, qc_filtered, args.run_order_column, args.outdir, basename)
