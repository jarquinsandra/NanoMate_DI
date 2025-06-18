import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from umap import UMAP


def strip_polarity_suffix(index):
    return index.str.replace("_pos", "", regex=False).str.replace("_neg", "", regex=False).str.strip().str.lower()

def plot_embedding(embedding, labels, title, out_path, explained_variance=None):
    plt.figure(figsize=(8, 5))
    if labels is not None and labels.nunique() > 1:
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="Set2", s=60, edgecolor="black", legend='full')
        plt.legend(bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0., fontsize='small', title=labels.name if hasattr(labels, 'name') else None)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=60, edgecolor="black")
        plt.text(0.95, 0.01, "Single group — no legend", transform=plt.gca().transAxes,
                 ha='right', va='bottom', fontsize=9, color='gray')
    if explained_variance is not None:
        plt.xlabel(f"PC1 ({explained_variance[0]:.1f}%)")
        plt.ylabel(f"PC2 ({explained_variance[1]:.1f}%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

warnings.filterwarnings("ignore", category=UserWarning)


def main(pos_path, neg_path, output_path, metadata_path=None, group_column=None,
         umap_n_neighbors=15, umap_min_dist=0.1, tsne_perplexity=30):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load aligned data
    pos_df = pd.read_csv(pos_path, index_col=0)
    neg_df = pd.read_csv(neg_path, index_col=0)

    # Normalize sample names
    pos_df.index = strip_polarity_suffix(pos_df.index.to_series())
    neg_df.index = strip_polarity_suffix(neg_df.index.to_series())

    # Find and match common samples
    common_samples = pos_df.index.intersection(neg_df.index)
    if common_samples.empty:
        raise ValueError("No common samples found between positive and negative data.")

    pos_df = pos_df.loc[common_samples].sort_index()
    neg_df = neg_df.loc[common_samples].sort_index()

    # Scale each block to unit variance
    scaler = StandardScaler(with_mean=False, with_std=True)
    pos_scaled = pd.DataFrame(scaler.fit_transform(pos_df), index=pos_df.index, columns=pos_df.columns)
    neg_scaled = pd.DataFrame(scaler.fit_transform(neg_df), index=neg_df.index, columns=neg_df.columns)

    # Concatenate and center
    combined = pd.concat([pos_scaled, neg_scaled], axis=1)
    combined_centered = combined - combined.mean(axis=0)
    combined_centered.to_csv(output_path)
    print(f"✅ Combined matrix saved to {output_path}")

    # Rescale for dimensionality reduction
    zscaler = StandardScaler()
    combined_scaled = zscaler.fit_transform(combined_centered)

    labels = None
    if metadata_path and group_column:
        meta = pd.read_csv(metadata_path, index_col=0)
        meta.index = strip_polarity_suffix(meta.index.to_series())
        missing_in_meta = sorted(set(common_samples) - set(meta.index))
        if missing_in_meta:
            print(f"⚠️ Warning: {len(missing_in_meta)} sample(s) in data not found in metadata. Skipping them: {missing_in_meta}")
        meta = meta.loc[meta.index.intersection(common_samples)]
        meta = meta.loc[combined_centered.index.intersection(meta.index)]
        labels = meta[group_column] if group_column in meta.columns else None

    # PCA
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(combined_scaled)
    explained = pca.explained_variance_ratio_ * 100
    plot_embedding(pca_emb, labels, "PCA", output_path.replace(".csv", "_PCA.png"), explained_variance=explained)

    # UMAP
    umap = UMAP(n_components=2, random_state=42, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)
    umap_emb = umap.fit_transform(combined_scaled)
    plot_embedding(umap_emb, labels, "UMAP", output_path.replace(".csv", "_UMAP.png"))

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
    tsne_emb = tsne.fit_transform(combined_scaled)
    plot_embedding(tsne_emb, labels, "t-SNE", output_path.replace(".csv", "_TSNE.png"))

    # Compute %RSD, median RSD, and % of features with RSD < 30% for QC and QA samples if available
    if metadata_path:
        for label in ["QC", "QA"]:
            if group_column:
                if label in meta[group_column].unique():
                    group_samples = meta[meta[group_column] == label].index
                    group_data = combined_centered.loc[group_samples]
                    rsd = group_data.std() / (group_data.mean().abs() + 1e-9) * 100
                    summary_df = pd.DataFrame({
                        "m/z": group_data.columns,
                        "%RSD": rsd.values,
                        "Below_30%": (rsd < 30).astype(int)
                    })
                    median_rsd = rsd.median()
                    summary_file = os.path.join(output_dir, f"{label}_rsd_summary.csv")
                    summary_df.to_csv(summary_file, index=False)
                    percent_below_30 = (rsd < 30).mean() * 100
                    summary_txt = os.path.join(output_dir, f"{label}_rsd_summary.txt")
                    with open(summary_txt, "w") as f:
                        f.write(f"{label} RSD Summary")
                        f.write(f"Number of features: {len(rsd)}")
                        f.write(f"Mean %RSD: {rsd.mean():.2f}%")
                        f.write(f"Median %RSD: {median_rsd:.2f}%")
                        f.write(f"Min %RSD: {rsd.min():.2f}%")
                        f.write(f"Max %RSD: {rsd.max():.2f}%")
                        f.write(f"Std of %RSD: {rsd.std():.2f}")
                        f.write(f"% Features with RSD < 30%: {percent_below_30:.1f}%")
                    print(f"📈 {label} RSD summary saved. Median %RSD: {median_rsd:.2f}%, Features <30% RSD: {percent_below_30:.1f}%")
                else:
                    print(f"⚠️ No '{label}' samples found in group column. Skipping RSD summary for {label}.")

    # Also generate PCA, UMAP, t-SNE without QC or QA samples
    if metadata_path and group_column:
        exclude_mask = ~meta[group_column].isin(["QC", "QA"])
        meta_no_qa_qc = meta[exclude_mask]
        combined_centered_no_qa_qc = combined_centered.loc[meta_no_qa_qc.index]
        labels_no_qa_qc = meta_no_qa_qc[group_column] if group_column in meta_no_qa_qc.columns else None
        combined_scaled_no_qa_qc = zscaler.fit_transform(combined_centered_no_qa_qc)

        pca_emb = pca.fit_transform(combined_scaled_no_qa_qc)
        explained = pca.explained_variance_ratio_ * 100
        plot_embedding(pca_emb, labels_no_qa_qc, "PCA (no QC/QA)", output_path.replace(".csv", "_noQCQA_PCA.png"), explained_variance=explained)

        umap_emb = umap.fit_transform(combined_scaled_no_qa_qc)
        plot_embedding(umap_emb, labels_no_qa_qc, "UMAP (no QC/QA)", output_path.replace(".csv", "_noQCQA_UMAP.png"))

        tsne_emb = tsne.fit_transform(combined_scaled_no_qa_qc)
        plot_embedding(tsne_emb, labels_no_qa_qc, "t-SNE (no QC/QA)", output_path.replace(".csv", "_noQCQA_TSNE.png"))

    print("📊 PCA, UMAP, and t-SNE plots saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine positive and negative aligned matrices with unit variance scaling and column centering, and perform PCA/UMAP/t-SNE visualization.")
    parser.add_argument("--pos", required=True, help="CSV file with aligned positive mode data")
    parser.add_argument("--neg", required=True, help="CSV file with aligned negative mode data")
    parser.add_argument("--output", required=True, help="Output CSV file for combined matrix")
    parser.add_argument("--metadata", help="Metadata CSV file for labeling samples")
    parser.add_argument("--group_column", help="Column in metadata to color plots by group")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="Number of neighbors for UMAP (default: 15)")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="Minimum distance for UMAP (default: 0.1)")
    parser.add_argument("--tsne_perplexity", type=float, default=30, help="Perplexity for t-SNE (default: 30)")
    args = parser.parse_args()

    main(args.pos, args.neg, args.output, args.metadata, args.group_column,
         args.umap_n_neighbors, args.umap_min_dist, args.tsne_perplexity)
