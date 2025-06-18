
import os
import argparse
from joblib import Parallel, delayed, dump
from pathlib import Path
import pymzml
import numpy as np
from tqdm import tqdm

CACHE_DIR = "peak_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Step 0: Pre-cache centroided peaks to speed up Step 1.")
parser.add_argument("--input_dir", required=True, help="Directory containing mzML files")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (default: all cores)")
args = parser.parse_args()

def cache_file_path(file_path):
    return os.path.join(CACHE_DIR, Path(file_path).stem + ".pkl")

def extract_peaks(file_path):
    cache_path = cache_file_path(file_path)
    if os.path.exists(cache_path):
        return f"✔️ Cached: {os.path.basename(file_path)}"
    try:
        run = pymzml.run.Reader(file_path)
        peaks = []
        for spectrum in run:
            if spectrum.ms_level == 1:
                peaks.extend(spectrum.peaks("centroided"))
        arr = np.array(peaks)
        dump(arr, cache_path)
        return f"✅ Cached: {os.path.basename(file_path)}"
    except Exception as e:
        return f"❌ Failed: {os.path.basename(file_path)} — {e}"

mzml_files = sorted([os.path.join(args.input_dir, f)
                     for f in os.listdir(args.input_dir) if f.lower().endswith(".mzml")])

print(f"🔍 Found {len(mzml_files)} mzML files. Caching peaks...")
results = Parallel(n_jobs=args.n_jobs)(delayed(extract_peaks)(f) for f in tqdm(mzml_files))
for r in results:
    print(r)
print("✅ Step 0 complete.")
