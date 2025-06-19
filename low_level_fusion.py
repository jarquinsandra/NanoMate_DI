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
        plt.legend(bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0., fontsize='small', title=labels.name)
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
         umap_n_neighbors=15, umap_min_dist=0.1, tsne_perplexity=30, only_plots=False):

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if only_plots:
        if not os.path.isfile(output_path):
            raise FileNotFoundError(f"Cannot find existing matrix to plot: {output_path}")
        combined_centered = pd.read_csv(output_path, index_col=0)
        combined_centered.index = (
            combined_centered.index.astype(str)
            .str.strip()
            .str.lower()
        )

        print(f"📂 Loaded existing combined matrix from {output_path}")
    else:
        pos_df = pd.read_csv(pos_path, index_col=0)
        neg_df = pd.read_csv(neg_path, index_col=0)

        pos_df.index = strip_polarity_suffix(pos_df.index.to_series())
        neg_df.index = strip_polarity_suffix(neg_df.index.to_series())

        common_samples = pos_df.index.intersection(neg_df.index)
        if common_samples.empty:
            raise ValueError("No common samples found between positive and negative data.")

        pos_df = pos_df.loc[common_samples].sort_index()
        neg_df = neg_df.loc[common_samples].sort_index()

        target_median = (pos_df.stack().median() + neg_df.stack().median()) / 2
        pos_scaled = pos_df - pos_df.stack().median() + target_median
        neg_scaled = neg_df - neg_df.stack().median() + target_median

        combined = pd.concat([pos_scaled, neg_scaled], axis=1)
        combined_centered = combined - combined.mean(axis=0)
        combined_centered.to_csv(output_path)
        print(f"✅ Combined matrix saved to {output_path}")

    zscaler = StandardScaler()
    combined_scaled = zscaler.fit_transform(combined_centered)

    labels = None
    meta = None
    if metadata_path and group_column:
        meta = pd.read_csv(metadata_path)
        meta["Sample"] = (
            meta["Sample"]
            .astype(str)
            .str.replace(".mzML", "", regex=False)
            .str.replace("_pos", "", regex=False)
            .str.replace("_neg", "", regex=False)
            .str.strip()
            .str.lower()
        )
        meta.set_index("Sample", inplace=True)

        # Align metadata and matrix by sample order
        meta = meta.loc[combined_centered.index.intersection(meta.index)]
        labels = meta.loc[combined_centered.index, group_column] if group_column in meta.columns else None

    suffix = f"groupby-{labels.name}" if labels is not None and hasattr(labels, 'name') else "no-groups"

    # Full-sample PCA/UMAP/t-SNE
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(combined_scaled)
    explained = pca.explained_variance_ratio_ * 100
    plot_embedding(pca_emb, labels, "PCA", output_path.replace(".csv", f"_PCA_{suffix}.png"), explained_variance=explained)

    umap = UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
    umap_emb = umap.fit_transform(combined_scaled)
    plot_embedding(umap_emb, labels, "UMAP", output_path.replace(".csv", f"_UMAP_{suffix}.png"))

    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
    tsne_emb = tsne.fit_transform(combined_scaled)
    plot_embedding(tsne_emb, labels, "t-SNE", output_path.replace(".csv", f"_TSNE_{suffix}.png"))

    # Plots excluding QC/QA
    if meta is not None and group_column in meta.columns:
        exclude_mask = ~meta[group_column].isin(["QC", "QA"])
        meta_no_qcqa = meta[exclude_mask]
        combined_no_qcqa = combined_centered.loc[meta_no_qcqa.index]
        labels_no_qcqa = meta_no_qcqa[group_column]

        combined_scaled_no_qcqa = zscaler.fit_transform(combined_no_qcqa)
        suffix_nq = f"groupby-{labels_no_qcqa.name}-noQCQA"

        pca_emb = pca.fit_transform(combined_scaled_no_qcqa)
        explained = pca.explained_variance_ratio_ * 100
        plot_embedding(pca_emb, labels_no_qcqa, "PCA (no QC/QA)", output_path.replace(".csv", f"_PCA_{suffix_nq}.png"), explained_variance=explained)

        umap_emb = umap.fit_transform(combined_scaled_no_qcqa)
        plot_embedding(umap_emb, labels_no_qcqa, "UMAP (no QC/QA)", output_path.replace(".csv", f"_UMAP_{suffix_nq}.png"))

        tsne_emb = tsne.fit_transform(combined_scaled_no_qcqa)
        plot_embedding(tsne_emb, labels_no_qcqa, "t-SNE (no QC/QA)", output_path.replace(".csv", f"_TSNE_{suffix_nq}.png"))

    print("📊 PCA, UMAP, and t-SNE plots saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine positive and negative matrices or regenerate plots.")
    parser.add_argument("--pos", help="CSV file with aligned positive mode data")
    parser.add_argument("--neg", help="CSV file with aligned negative mode data")
    parser.add_argument("--output", required=True, help="Output CSV file (e.g., fused.csv)")
    parser.add_argument("--outdir", default="fusion_output", help="Directory to save outputs (default: fusion_output)")
    parser.add_argument("--metadata", help="Metadata CSV file for labeling samples")
    parser.add_argument("--group_column", help="Column in metadata to color plots")
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--tsne_perplexity", type=float, default=30)
    parser.add_argument("--only_plots", action="store_true", help="Use existing matrix to generate plots only")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    output_file = os.path.join(args.outdir, os.path.basename(args.output))

    main(args.pos, args.neg, output_file, args.metadata, args.group_column,
         args.umap_n_neighbors, args.umap_min_dist, args.tsne_perplexity,
         only_plots=args.only_plots)
