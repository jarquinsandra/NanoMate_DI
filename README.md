
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
