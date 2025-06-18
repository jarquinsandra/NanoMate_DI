# NanoMate Metabolomics Pipeline

This repository contains a set of scripts used in a NanoMate-based untargeted metabolomics data analysis pipeline. It includes RAW data conversion, peak alignment, m/z drift correction, QC-based filtering, normalization, imputation, and low-level data fusion for downstream visualization and statistics.

---

## 🔧 Scripts Overview

| Script | Purpose |
|--------|---------|
| `msconvert_wrapper.py` | Converts Thermo RAW files to centroided mzML format split by polarity using ProteoWizard MSConvert. |
| `Metadata_improvement.py` | Merges injection order timestamps with sample metadata. |
| `step_0_cache_peaks.py` | Extracts and caches centroided MS1 peaks from mzML files to speed up downstream processing. |
| `Drift_correction.py` | Applies m/z drift correction using anchor peaks from pooled QC samples. |
| `step_1_peak_alignment.py` | Aligns peaks across samples by binning to a common m/z axis. Supports optional background subtraction and m/z correction input. |
| `Step_2_filter.py` | Filters features, applies normalization (PQN, TIC, median), QC-RLSC correction, ComBat batch correction, and imputes missing values. |
| `QC_evaluation_2.py` | Evaluates data quality using PCA, UMAP, t-SNE, Mahalanobis distance, and QC RSD. |
| `low_level_fusion.py` | Combines positive and negative mode data with PCA/UMAP/t-SNE visualization and RSD summaries. |

---

## 📦 Installation

To install dependencies, use:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Each script includes command-line help. For example:

```bash
python step_1_peak_alignment.py --help
```

---

## 📁 Folder Structure

```text
.
├── msconvert_wrapper.py
├── Metadata_improvement.py
├── step_0_cache_peaks.py
├── Drift_correction.py
├── step_1_peak_alignment.py
├── Step_2_filter.py
├── QC_evaluation_2.py
├── low_level_fusion.py
├── requirements.txt
└── README.md
```

---

## 📞 Contact

For questions, please reach out to the script maintainer.

---

# NanoMate Script Options

## `msconvert_wrapper.py`

**Description:** Convert Thermo RAW files to centroided mzML split by polarity using MSConvert.


**Options:**

- `-i / --input`: Input directory with RAW files.
- `-o / --output`: Output base directory.

## `Metadata_improvement.py`

**Description:** Merge mzML-derived injection timestamps with sample metadata.


**Options:**

- `--input_dir`: Folder containing mzML files.
- `--metadata`: CSV metadata file with 'Sample' column.
- `--output`: Output CSV filename.
- `--plate_location`: Optional CSV file with sample-to-plate mapping.
- `--extra_info`: Optional second CSV file to merge based on sample base name.

## `step_0_cache_peaks.py`

**Description:** Cache centroided MS1 peaks to speed up downstream steps.


**Options:**

- `--input_dir`: Directory containing mzML files.
- `--n_jobs`: Number of parallel jobs (default: all cores).

## `Drift_correction.py`

**Description:** Apply m/z drift correction using pooled QC samples and anchor peaks.


**Options:**

- `--input_dir`: Directory with mzML files.
- `--metadata`: Metadata CSV file.
- `--qc_label`: Label to identify QC samples.
- `--sample_column`: Column with sample names (default: Sample).
- `--sample_type_column`: Column identifying sample types (default: Type).
- `--polarity_column`: Column for polarity (e.g. positive/negative).
- `--polarity`: Polarity to process (positive or negative).
- `--output_dir`: Directory to save corrected files.
- `--top_n`: Number of top m/z anchors to use.
- `--strict_qc_filtering`: Enable stricter QC selection.
- `--save_csv`: Save corrected peaks as CSV.
- `--plot`: Generate drift and match summary plots.
- `--n_jobs`: Number of parallel jobs.

## `step_1_peak_alignment.py`

**Description:** Align and bin centroided peaks across all samples.


**Options:**

- `--input_dir`: Directory with mzML files.
- `--background`: Optional background mzML file.
- `--no_background_subtraction`: Skip background subtraction.
- `--polarity`: Polarity (positive/negative).
- `--aggregate`: Intensity aggregation method (sum/mean).
- `--sn_threshold`: Signal-to-noise ratio threshold.
- `--min_intensity`: Minimum intensity threshold.
- `--output_file`: Name of output CSV file.
- `--metadata`: Metadata file for matching and filtering.
- `--group_column`: Metadata column for sample grouping.
- `--ppm`: m/z tolerance in ppm.
- `--adaptive_binning`: Use KDE-based adaptive m/z binning.
- `--adaptive_method`: Binning method (merge/kde).
- `--kde_sample_size`: Number of m/z values to sample for KDE.
- `--kde_seed`: Random seed for KDE.
- `--n_jobs`: Number of parallel jobs.
- `--load_corrected_peaks`: Folder with JSON drift-corrected peaks.
- `--drift_summary`: Path to drift_summary.csv to filter corrected samples.

## `Step_2_filter.py`

**Description:** Filter, normalize, batch-correct, and impute peak data.


**Options:**

- `--input`: Aligned peak table CSV.
- `--metadata`: Metadata CSV file.
- `--group_column`: Column identifying sample groups (e.g. QC/QA).
- `--batch_column`: Column for ComBat batch correction.
- `--mahal_threshold`: Mahalanobis distance threshold for outlier filtering.
- `--neighbors`: Number of neighbors for kNN imputation.
- `--impute_method`: Imputation method: knn or rf.
- `--zero_threshold`: Max % of zeros allowed per feature (default: 0.8).
- `--log`: Log transformation: log2, log10, ln.
- `--plot_format`: png or pdf (default: png).
- `--normalisation`: none, PQN, TIC, or median (default: PQN).
- `--qc_rlsc`: Apply QC-RLSC correction using LOESS.
- `--qc_label`: Label to identify QC samples (default: QC).
- `--run_order_column`: Run order column in metadata.
- `--outdir`: Directory to save outputs.
- `--skip-plots`: Skip generation of plots.
- `--verbose`: Print more detailed logs.

## `QC_evaluation_2.py`

**Description:** Evaluate sample quality using PCA, UMAP, t-SNE, and Mahalanobis filtering.


**Options:**

- `--input`: Aligned peak table.
- `--metadata`: Metadata CSV file.
- `--sample_type_column`: Column labeling sample type.
- `--qc_label`: Label for QC samples.
- `--qa_label`: Label for QA samples (optional).
- `--plate_column`: Metadata column to color by (e.g., plate).
- `--outdir`: Directory to save output (default: qc_evaluation_results).
- `--mahal_threshold`: Mahalanobis distance threshold.
- `--export_cleaned_data`: Export cleaned tables.
- `--pca_batch_size`: Batch size for Incremental PCA.
- `--subset_sample_count`: Number of non-QC samples to randomly use.
- `--random_seed`: Random seed (default: 42).
- `--plot_qc_drift`: Plot intensity drift of top QC features.
- `--run_order_column`: Column in metadata for injection order.
- `--no_filtering`: Skip outlier removal.
- `--log_transform`: Apply log2(x+1) transformation.
- `--plot_all`: Force plotting all samples (>300).
- `--tsne_perplexity`: Perplexity for t-SNE.
- `--umap_neighbors`: Neighbors for UMAP.
- `--umap_min_dist`: Minimum distance for UMAP.
- `--exclude_sample_types`: List of sample types to exclude.
- `--verbose`: Print logs.

## `low_level_fusion.py`

**Description:** Combine positive and negative mode peak tables and visualize with PCA, UMAP, t-SNE.


**Options:**

- `--pos`: Aligned positive mode CSV file.
- `--neg`: Aligned negative mode CSV file.
- `--output`: Output CSV file for combined data.
- `--metadata`: Optional metadata CSV.
- `--group_column`: Column in metadata for coloring samples.
- `--umap_n_neighbors`: UMAP: number of neighbors (default: 15).
- `--umap_min_dist`: UMAP: minimum distance (default: 0.1).
- `--tsne_perplexity`: t-SNE perplexity (default: 30).