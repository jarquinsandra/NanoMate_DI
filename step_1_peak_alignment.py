import argparse
import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from time import perf_counter
from joblib import Parallel, delayed, cpu_count, dump, load
from scipy.sparse import csr_matrix
from pathlib import Path
import logging
import pymzml
from numba import njit
import warnings

# Suppress joblib/loky resource_tracker PermissionError warnings on Windows
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')
warnings.filterwarnings("ignore", message="resource_tracker:.*used by another process")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
EPSILON = 1e-6
CACHE_DIR = "peak_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@njit
def fast_bin_to_axis(peaks, mz_axis, ppm, method_sum=True):
    intensity = np.zeros(len(mz_axis))
    for i in range(peaks.shape[0]):
        mz, inten = peaks[i]
        for j in range(mz_axis.shape[0]):
            tol = mz_axis[j] * ppm * 1e-6
            if abs(mz_axis[j] - mz) <= tol:
                if method_sum:
                    intensity[j] += inten
                else:
                    intensity[j] = (intensity[j] + inten) / 2 if intensity[j] > 0 else inten
                break
    return intensity

def bin_sample_to_axis_sparse(peaks, mz_axis, ppm, method="sum"):
    method_sum = method == "sum"
    binned = fast_bin_to_axis(np.array(peaks), np.array(mz_axis), ppm, method_sum)
    return csr_matrix(binned)

def validate_metadata(metadata_path, input_dir, drift_summary_path=None):
    df = pd.read_csv(metadata_path)
    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace(".mzML", "", case=False, regex=False).str.strip().str.lower()
    df.set_index(df.columns[0], inplace=True)

    mzml_files = [os.path.splitext(f)[0].lower() for f in os.listdir(input_dir) if f.lower().endswith(".mzml")]
    unmatched = set(df.index) - set(mzml_files)
    if unmatched:
        logging.warning(f"{len(unmatched)} metadata sample(s) not found in .mzML files:")
        for name in sorted(unmatched):
            logging.warning(f"   - {name}")

    if drift_summary_path:
        try:
            drift_df = pd.read_csv(drift_summary_path)
            drift_df.columns = [c.strip().replace('\ufeff', '') for c in drift_df.columns]
            drift_df["Sample"] = drift_df["Sample"].astype(str).str.replace(".mzML", "", case=False, regex=False).str.strip().str.lower()
            if any(col.lower() == "corrected" for col in drift_df.columns):
                corrected_col = next(col for col in drift_df.columns if col.lower() == "corrected")
                drift_df[corrected_col] = drift_df[corrected_col].astype(str).str.lower().isin(["true", "1", "yes"])
                corrected = drift_df.loc[drift_df[corrected_col], "Sample"].tolist()
                df = df.loc[df.index.intersection(corrected)]
                logging.info(f"📌 Filtered metadata to {len(df)} corrected samples from drift summary.")
            else:
                logging.warning("⚠️ 'Corrected' column not found in drift_summary.csv. Using all listed samples.")
                corrected = drift_df["Sample"].tolist()
                df = df.loc[df.index.intersection(corrected)]
        except Exception as e:
            logging.error(f"❌ Error reading drift summary: {e}")
            raise e
    return df

def load_centroids(file_path):
    cache_file = os.path.join(CACHE_DIR, Path(file_path).stem + ".pkl")
    if os.path.isfile(cache_file):
        return load(cache_file)
    run = pymzml.run.Reader(file_path)
    peaks = []
    for spectrum in run:
        if spectrum.ms_level == 1:
            peaks.extend(spectrum.peaks("centroided"))
    arr = np.array(peaks)
    dump(arr, cache_file)
    return arr

def parallel_load_peaks(input_dir, sample_files, cores):
    def _load(name):
        return name, load_centroids(os.path.join(input_dir, name))
    return dict(Parallel(n_jobs=cores)(delayed(_load)(f) for f in sample_files))

def merge_centroids(all_peaks, ppm):
    all_peaks = sorted(all_peaks, key=lambda x: x[0])
    clusters, start = [], 0
    while start < len(all_peaks):
        ref = all_peaks[start][0]
        tol = ref * ppm * 1e-6
        cluster = [all_peaks[start]]
        i = start + 1
        while i < len(all_peaks) and abs(all_peaks[i][0] - ref) <= tol:
            cluster.append(all_peaks[i])
            i += 1
        clusters.append(cluster)
        start = i
    return [np.mean([mz for mz, _ in cluster]) for cluster in clusters]

def kde_adaptive_binning(mz_array, bandwidth=0.01, base_bin_width=0.01, sample_size=10000, seed=None):
    from sklearn.neighbors import KernelDensity
    if seed is not None:
        np.random.seed(seed)
    sample = mz_array if len(mz_array) <= sample_size else np.random.choice(mz_array, sample_size, replace=False)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(sample[:, None])
    density = np.exp(kde.score_samples(mz_array[:, None]))
    norm_density = density / density.max()
    widths = base_bin_width + (1 - norm_density) * 0.1
    edges = np.cumsum(widths) - np.cumsum(widths)[0] + mz_array.min()
    bins = [mz_array[(mz_array >= edges[i]) & (mz_array < edges[i+1])].mean()
            for i in range(len(edges) - 1) if any((mz_array >= edges[i]) & (mz_array < edges[i+1]))]
    return np.unique(np.round(bins, 5))

def save_top_peaks(df, outdir, n=5, rounding=5):
    df.columns = df.columns.round(rounding)
    top = df.max().sort_values(ascending=False).head(n)
    pd.DataFrame({"m/z": top.index, "MaxIntensity": top.values}).to_csv(
        f"{outdir}/top_peaks_filtered_table.csv", index=False
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--background")
    parser.add_argument("--no_background_subtraction", action="store_true")
    parser.add_argument("--polarity", choices=["positive", "negative"], required=True)
    parser.add_argument("--aggregate", choices=["sum", "mean"], default="mean")
    parser.add_argument("--sn_threshold", type=float, default=3.0)
    parser.add_argument("--min_intensity", type=float, default=0)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--metadata")
    parser.add_argument("--group_column")
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--mz_decimals", type=int, default=5)
    parser.add_argument("--adaptive_binning", action="store_true")
    parser.add_argument("--adaptive_method", choices=["merge", "kde"], default="merge")
    parser.add_argument("--kde_sample_size", type=int, default=10000)
    parser.add_argument("--kde_seed", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--load_corrected_peaks")
    parser.add_argument("--drift_summary")
    args = parser.parse_args()
    sample_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".mzml"))
    cores = args.n_jobs if args.n_jobs > 0 else cpu_count()
    
    logging.info("🚀 Step 1 started: Aligning peaks with the following settings:")
    logging.info(json.dumps(vars(args), indent=2))
    
    output_dir = "step1_results"
    os.makedirs(output_dir, exist_ok=True)
    start = perf_counter()
    
    if args.metadata:
        logging.info("📄 Validating metadata...")
        metadata_df = validate_metadata(args.metadata, args.input_dir, args.drift_summary)
        logging.info(f"✅ Metadata loaded with {len(metadata_df)} samples.")
    else:
        metadata_df = None
        logging.info("⚠️ No metadata provided.")
    
    sample_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".mzml"))
    logging.info(f"📦 Found {len(sample_files)} mzML files in {args.input_dir}")
    cores = args.n_jobs if args.n_jobs > 0 else cpu_count()
    
    logging.info("🔄 Loading centroided peaks from files...")
    t1 = perf_counter()
    
    def load_corrected_json_peaks(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return np.array(data)
    
    if args.load_corrected_peaks:
        logging.info("📦 Loading corrected peaks from JSON files...")
        def _load_corrected(name):
            json_file = os.path.join(args.load_corrected_peaks, name.replace(".mzML", "") + ".json")
            if not os.path.isfile(json_file):
                logging.warning(f"⚠️ Skipping {name} — JSON file not found: {json_file}")
                return name, None
            return name, load_corrected_json_peaks(json_file)

        sample_peaks_raw = Parallel(n_jobs=cores)(
            delayed(_load_corrected)(f) for f in sample_files
        )
        sample_peaks = {name: peaks for name, peaks in sample_peaks_raw if peaks is not None}

    else:
        logging.info("🔄 Loading centroided peaks from .mzML files...")
        sample_peaks = parallel_load_peaks(args.input_dir, sample_files, cores)
    
    logging.info(f"✅ Finished loading peaks in {perf_counter() - t1:.2f} seconds.")
    
    all_peaks = np.concatenate(list(sample_peaks.values()))
    mz_array = np.array([mz for mz, _ in all_peaks])
    logging.info("🧠 Building m/z axis...")
    mz_axis = (
        kde_adaptive_binning(mz_array, sample_size=args.kde_sample_size, seed=args.kde_seed)
        if args.adaptive_method == "kde"
        else np.round(merge_centroids(all_peaks, args.ppm), args.mz_decimals)
    )
    
    logging.info("📊 Binning all samples to common m/z axis...")
    t2 = perf_counter()
    binned = Parallel(n_jobs=cores)(
    delayed(lambda f: bin_sample_to_axis_sparse(sample_peaks[f], mz_axis, args.ppm, args.aggregate))(f)
    for f in tqdm(sample_peaks.keys(), desc="Binning")  
    )

    logging.info(f"✅ Finished binning in {perf_counter() - t2:.2f} seconds.")
    
    df = pd.DataFrame(
    np.vstack([b.toarray().flatten() for b in binned]),
    index=list(sample_peaks.keys()),
    columns=np.round(mz_axis, args.mz_decimals)
    )

    
    if args.background and not args.no_background_subtraction:
        logging.info("➖ Performing background subtraction...")
        bg_peaks = load_centroids(args.background)
        bg_intens = bin_sample_to_axis_sparse(bg_peaks, mz_axis, args.ppm, args.aggregate).toarray().flatten()
        df = df.subtract(bg_intens, axis=1).clip(lower=0)
    
    logging.info("🔍 Filtering peaks by S/N and intensity thresholds...")
    t3 = perf_counter()
    df_filtered = df.copy()
    if args.sn_threshold > 0 and args.background:
        sn = df / (bg_intens + EPSILON)
        df_filtered = df.loc[:, sn.max() >= args.sn_threshold]
    if args.min_intensity > 0:
        df_filtered = df_filtered.loc[:, df_filtered.max() >= args.min_intensity]
    logging.info(f"✅ Finished filtering in {perf_counter() - t3:.2f} seconds.")
    
    final_path = os.path.join(output_dir, args.output_file or f"aligned_peaks_{args.polarity}_ppm.csv")
    logging.info(f"💾 Saving aligned peak matrix to: {final_path}")
    df_filtered.to_csv(final_path)
    save_top_peaks(df_filtered, output_dir)
    logging.info("✅ Saved top filtered peaks.")
    logging.info(f"⏱️ Total runtime: {perf_counter() - start:.2f} seconds.")
    
if __name__ == "__main__":
    main()
