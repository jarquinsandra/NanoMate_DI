
NanoMate Preprocessing Quick Start (No Metadata)

This pipeline includes two main steps to preprocess and normalize mzML files without needing metadata or QC samples.

---

Step 1: Align peaks from mzML files

Replace `path/to/mzml_files` with the folder containing your .mzML files.

```bash
python step_1_peak_alignment.py   --input_dir path/to/mzml_files   --polarity positive   --output_file aligned_peaks.csv
```

Optional parameters:
- `--sn_threshold 3.0` : Filter low-intensity peaks based on S/N.
- `--adaptive_binning` : Use density-based binning for improved alignment.
- `--n_jobs -1` : Use all CPU cores for faster processing.

---

Step 2: Normalize and impute missing values

```bash
python Step_2_filter.py   --input aligned_peaks.csv   --normalisation median   --impute_method knn   --outdir step2_output   --skip-plots
```

The final cleaned matrix will be saved to:
`step2_output/aligned_peaks_filtered_imputed.csv`

---

Install dependencies with:

```bash
pip install -r requirements.txt
```

Happy preprocessing!


---

## 📦 Installing Python Dependencies

Before running the scripts, make sure you have Python 3.7 or later installed.

To install the required Python packages, run:

```bash
pip install -r requirements.txt
```

If you're using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This ensures you have all the necessary libraries for running the preprocessing pipeline.
