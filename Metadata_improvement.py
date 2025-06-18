from pathlib import Path
import pandas as pd
import argparse
import unicodedata

def clean_sample(s):
    """Normalize sample name: lowercase, remove .mzML, strip, ASCII only."""
    return unicodedata.normalize("NFKD", str(s)) \
        .encode("ascii", "ignore").decode() \
        .replace(".mzML", "").strip().lower()
def strip_polarity(s):
    """Remove _pos or _neg suffixes from sample names for matching."""
    return s.replace("_pos", "").replace("_neg", "")


def extract_run_timestamp(filepath):
    """Extract timestamp from mzML header line."""
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                if "startTimeStamp=" in line:
                    parts = line.strip().split("startTimeStamp=")
                    if len(parts) > 1:
                        timestamp = parts[1].split('"')[1]
                        print(f"✔ Found timestamp in {filepath.name}: {timestamp}")
                        return timestamp
    except Exception as e:
        print(f"❌ Error reading {filepath.name}: {e}")
    return None

def main(mzml_dir, metadata_path, output_path, plate_location=None, extra_info=None):

    mzml_dir = Path(mzml_dir)
    metadata_path = Path(metadata_path)
    records = []

    # Extract timestamps
    for file in sorted(mzml_dir.glob("*.mzML")):
        ts = extract_run_timestamp(file)
        if ts:
            records.append({"Sample": file.stem, "StartTimeStamp": ts})

    if not records:
        print("⚠️ No valid timestamps found.")
        return

    df_order = pd.DataFrame(records)
    df_order["StartTimeStamp"] = pd.to_datetime(df_order["StartTimeStamp"])
    df_order = df_order.sort_values("StartTimeStamp").reset_index(drop=True)
    df_order["RunOrder"] = df_order.index + 1

    # Load metadata
    df_meta = pd.read_csv(metadata_path)
    if "Sample" not in df_meta.columns:
        print("❌ Metadata must contain a 'Sample' column.")
        return

    # Normalize sample names
    df_meta["Sample"] = df_meta["Sample"].astype(str).apply(clean_sample)
    df_order["Sample"] = df_order["Sample"].astype(str).apply(clean_sample)

    # Debug previews
    print("\n📋 Sample names from mzML:")
    print(df_order["Sample"].tolist()[:5])
    print("\n📋 Sample names from metadata:")
    print(df_meta["Sample"].tolist()[:5])
    print("✅ Sample names match check:")
    print("mzML unique:", len(df_order["Sample"].unique()))
    print("metadata unique:", len(df_meta["Sample"].unique()))

    unmatched = sorted(set(df_order["Sample"]) - set(df_meta["Sample"]))
    print(f"❌ Unmatched samples in mzML: {len(unmatched)}")
    for u in unmatched[:10]:
        print(f" - {repr(u)}")

    
    # Load plate location
    def merge_by_base_sample(df_base, merge_path, label):
        if merge_path is None:
            print(f"⚠️ No {label} file provided. Skipping {label} integration.")
            return df_base
        merge_path = Path(merge_path)
        if not merge_path.exists():
            print(f"❌ {label} file not found: {merge_path}")
            return df_base
        df_add = pd.read_csv(merge_path)
        if "Sample" not in df_add.columns:
            print(f"❌ {label} file must have a 'Sample' column.")
            return df_base
        df_add["BaseSample"] = df_add["Sample"].astype(str).apply(clean_sample).apply(strip_polarity)
        df_base["BaseSample"] = df_base["Sample"].apply(strip_polarity)
        df_base = df_base.merge(df_add.drop(columns=["Sample"]), on="BaseSample", how="left")
        print(f"✅ {label} information merged based on base sample names.")
        return df_base

    df_meta = merge_by_base_sample(df_meta, plate_location, "plate_location")
    df_meta = merge_by_base_sample(df_meta, extra_info, "extra_info")
    df_meta.drop(columns=["BaseSample"], inplace=True, errors="ignore")
    

    # Merge
    df_merged = df_order.merge(df_meta, on="Sample", how="left")

    # Check for real merge failures (where *all* metadata fields failed)
    meta_cols = [col for col in df_meta.columns if col != "Sample"]
    missing = df_merged[df_merged[meta_cols].isnull().all(axis=1)]
    if not missing.empty:
        for s in missing["Sample"]:
            print(f"⚠️ Warning: Sample '{s}' found in mzML but missing in metadata.")

    # Drop only those truly unmatched rows
    df_merged = df_merged[~df_merged[meta_cols].isnull().all(axis=1)]

    # Save result
    df_merged.to_csv(output_path, index=False)
    print(f"✅ Merged metadata with run order saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate run order from mzML and merge with metadata.")
    parser.add_argument("--plate_location", help="Optional CSV file with sample-to-plate mapping.")
    parser.add_argument("--extra_info", help="Optional second CSV file to merge based on sample base name.")
    parser.add_argument("--input_dir", required=True, help="Folder containing mzML files.")
    parser.add_argument("--metadata", required=True, help="CSV metadata file with 'Sample' column.")
    parser.add_argument("--output", required=True, help="Output CSV filename.")
    args = parser.parse_args()

    main(args.input_dir, args.metadata, args.output, args.plate_location, args.extra_info)