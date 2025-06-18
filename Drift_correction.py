
import os
import argparse
import numpy as np
import pandas as pd
import pymzml
import json
import logging
from collections import defaultdict
from scipy.stats import linregress
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load, Parallel, delayed

# ------------------ Setup ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
CACHE_DIR = Path("peak_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description="Apply m/z drift correction using pooled QC samples.")
parser.add_argument("--input_dir", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--qc_label", required=True)
parser.add_argument("--sample_column", default="Sample")
parser.add_argument("--sample_type_column", default="Type")
parser.add_argument("--polarity_column", default="Polarity")
parser.add_argument("--polarity", choices=["positive", "negative"], required=True)
parser.add_argument("--output_dir", default="corrected_peaks")
parser.add_argument("--top_n", type=int, default=10)
parser.add_argument("--strict_qc_filtering", action="store_true")
parser.add_argument("--save_csv", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--n_jobs", type=int, default=1)
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
# ------------------ Map lowercase sample name to actual filename ------------------
file_map = {
    f.stem.lower(): f.name
    for f in Path(args.input_dir).glob("*.mzML")
}

# ------------------ Load metadata ------------------
metadata = pd.read_csv(args.metadata)
required_cols = [args.sample_column, args.sample_type_column, args.polarity_column]
for col in required_cols:
    if col not in metadata.columns:
        raise ValueError(f"Missing column '{col}' in metadata.")

metadata[args.sample_column] = (
    metadata[args.sample_column]
    .astype(str)
    .str.replace(".mzML", "", regex=False)
    .str.strip()
)
metadata[args.sample_type_column] = metadata[args.sample_type_column].astype(str).str.strip().str.upper()
metadata[args.polarity_column] = metadata[args.polarity_column].astype(str).str.strip().str.lower()
metadata = metadata[metadata[args.polarity_column] == args.polarity.lower()]

qc_files = metadata[metadata[args.sample_type_column] == args.qc_label.upper()][args.sample_column].tolist()
all_files = metadata[args.sample_column].tolist()
qc_files = [f.strip() for f in qc_files]
all_files = [f.strip() for f in all_files]


# ------------------ Extract peaks ------------------
def extract_and_cache_peaks(fname):
    actual_name = file_map.get(fname.lower())
    if not actual_name:
        logging.warning(f"File not found: {args.input_dir}/{fname}.mzML")
        return fname, None
    filepath = Path(args.input_dir) / actual_name
    cache_file = CACHE_DIR / (fname.lower() + ".pkl")
    if cache_file.exists():
        return fname, load(cache_file)
    try:
        run = pymzml.run.Reader(str(filepath))
        peaks = []
        for spectrum in run:
            if spectrum.ms_level == 1:
                peaks.extend(spectrum.peaks("centroided"))
        arr = np.array(peaks)
        dump(arr, cache_file)
        return fname, arr
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return fname, None

logging.info("🔍 Loading centroid data...")
results = Parallel(n_jobs=args.n_jobs)(delayed(extract_and_cache_peaks)(fname) for fname in all_files)
peak_map = {fname: peaks for fname, peaks in results if peaks is not None}
existing_files = list(peak_map.keys())

# ------------------ Initial QC filtering ------------------
filtered_qc_files = [f for f in qc_files if f in peak_map]
if not filtered_qc_files:
    logging.error("❌ No valid QC files loaded — check your input directory and metadata.")
    exit(1)


# ------------------ Anchor Calculation ------------------
def calculate_anchors(qc_files, spacing=0.01):
    all_qc_peaks = np.concatenate([peak_map[f] for f in qc_files])
    bins = np.round(all_qc_peaks[:, 0], 5)
    intensities = defaultdict(float)
    for mz, inten in zip(bins, all_qc_peaks[:, 1]):
        intensities[mz] += inten
    top_mz = sorted(intensities.items(), key=lambda x: x[1], reverse=True)
    selected = []
    for mz, inten in top_mz:
        if all(abs(mz - s) > spacing for s in selected):
            selected.append(mz)
        if len(selected) == args.top_n:
            break
    return np.array(selected)

anchors = calculate_anchors(filtered_qc_files)

# ------------------ Drift Correction ------------------
def find_shifts(peaks, anchors, ppm=10):
    matched = []
    for ref in anchors:
        tol = ref * ppm * 1e-6
        nearby = peaks[(peaks[:, 0] >= ref - tol) & (peaks[:, 0] <= ref + tol)]
        if len(nearby) > 0:
            closest = nearby[np.argmin(np.abs(nearby[:, 0] - ref))][0]
            matched.append((ref, closest))
    return matched

drift_data = []
match_counts = []
match_log = []

logging.info("🛠️  Estimating and applying drift corrections...")
for fname in existing_files:
    peaks = peak_map[fname]
    matched = find_shifts(peaks, anchors)
    match_counts.append((fname, len(matched)))
    match_log.append({"Sample": fname, "Matched": len(matched)})

    if len(matched) < 3:
        corrected = peaks
        slope, intercept = np.nan, np.nan
        logging.warning(f"⚠️ Not enough matches in {fname} — skipping correction.")
    else:
        ref_vals, observed_vals = zip(*matched)
        slope, intercept, *_ = linregress(observed_vals, ref_vals)
        corrected_mz = slope * peaks[:, 0] + intercept
        corrected = np.column_stack([corrected_mz, peaks[:, 1]])
        logging.info(f"📈 {fname}: slope={slope:.8f}, intercept={intercept:.6f}, n={len(matched)}")

    json.dump(corrected.tolist(), open(Path(args.output_dir) / (fname + ".json"), "w"))
    if args.save_csv:
        pd.DataFrame(corrected, columns=["m/z", "Intensity"]).to_csv(Path(args.output_dir) / (fname + ".csv"), index=False)

    sample_type = metadata.loc[metadata[args.sample_column].str.lower() == fname.lower(), args.sample_type_column]
    if sample_type.empty:
        logging.warning(f"⚠️ Sample '{fname}' not found in metadata. Skipping type assignment.")
        sample_type = "Unknown"
    else:
        sample_type = sample_type.values[0]

    corrected_flag = not np.isnan(slope) and not np.isnan(intercept)
    drift_data.append((fname, slope, intercept, sample_type, corrected_flag))

# ------------------ Optional strict QC filtering ------------------
df_drift = pd.DataFrame(drift_data, columns=["Sample", "Slope", "Intercept", "Type", "Corrected"])
if args.strict_qc_filtering:
    qc_drift = df_drift[(df_drift["Type"] == "QC") &
                        (df_drift["Intercept"].abs() < 5e-6) &
                        ((df_drift["Slope"] - 1).abs() < 1e-6)]
    filtered_qc_files = [f for f in qc_drift["Sample"] if f in peak_map]
    if len(filtered_qc_files) < 2:
        logging.warning("Too few high-quality QCs. Falling back to all QCs.")
        filtered_qc_files = [f for f in qc_files if f in peak_map]
    else:
        logging.info(f"Using {len(filtered_qc_files)} high-quality QCs.")
        anchors = calculate_anchors(filtered_qc_files)

# ------------------ Save stats ------------------
df_drift.to_csv(Path(args.output_dir) / "drift_summary.csv", index=False)
pd.DataFrame(match_log).to_csv(Path(args.output_dir) / "anchor_match_counts.csv", index=False)

# ------------------ Plot (optional) ------------------
if args.plot:
    plt.figure(figsize=(8, 5))
    dfm = pd.DataFrame(match_counts, columns=["Sample", "MatchedAnchors"])
    sns.histplot(dfm["MatchedAnchors"], bins=range(0, args.top_n + 2))
    plt.axvline(3, color="red", linestyle="--", label="Min required")
    plt.title("Anchor Peak Matches per Sample")
    plt.xlabel("Matched Anchors")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "anchor_match_histogram.png")
    plt.close()

    df_drift["Intercept_ppm"] = df_drift["Intercept"] * 1e6
    df_drift["Slope_ppm"] = (df_drift["Slope"] - 1) * 1e6
    df_drift["Outlier"] = (df_drift["Intercept_ppm"].abs() > 5) | (df_drift["Slope_ppm"].abs() > 1)

    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    sns.lineplot(data=df_drift, x="Sample", y="Intercept_ppm", hue="Type", marker="o", ax=ax1)
    ax1.set_ylabel("Intercept Drift (ppm)", color="tab:blue")
    ax1.axhline(y=5, color='blue', linestyle=':', linewidth=1)
    ax1.axhline(y=-5, color='blue', linestyle=':', linewidth=1)
    for _, row in df_drift[df_drift["Outlier"]].iterrows():
        ax1.text(row["Sample"], row["Intercept_ppm"], row["Sample"], color="blue", fontsize=8, rotation=45, ha="center")

    ax2 = ax1.twinx()
    ax2.plot(df_drift["Sample"], df_drift["Slope_ppm"], color="tab:red", marker="x", linestyle="--")
    ax2.set_ylabel("Slope Drift (ppm)", color="tab:red")
    ax2.axhline(y=1, color='red', linestyle=':', linewidth=1)
    ax2.axhline(y=-1, color='red', linestyle=':', linewidth=1)
    for _, row in df_drift[df_drift["Outlier"]].iterrows():
        ax2.text(row["Sample"], row["Slope_ppm"], row["Sample"], color="red", fontsize=8, rotation=45, ha="center")

    plt.xticks(rotation=90)
    plt.title("Instrument Drift with Outliers")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "instrument_drift_plot.png")
    plt.close()
