import os
import subprocess
import argparse
from pathlib import Path

# Full path to msconvert.exe
MSC_PATH = r"C:\Program Files\ProteoWizard\ProteoWizard 3.0.25132.2257859\msconvert.exe"

def run_msconvert(input_file, output_dir, polarity):
    """Run MSConvert with polarity filter and vendor peak picking."""
    if polarity == "positive":
        polarity_filter = "polarity positive"
        suffix = "_pos"
    elif polarity == "negative":
        polarity_filter = "polarity negative"
        suffix = "_neg"
    else:
        raise ValueError("Polarity must be 'positive' or 'negative'.")

    filename_stem = Path(input_file).stem + suffix + ".mzML"
    output_path = Path(output_dir) / filename_stem

    command = [
        MSC_PATH,
        input_file,
        "--mzML",
        "--filter", "peakPicking true 1-",
        "--filter", polarity_filter,
        "-o", str(output_dir),
        "--outfile", filename_stem
    ]

    print(f"Converting {input_file} -> {output_path}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file} ({polarity}): {e}")

def convert_all(input_dir, output_base):
    input_dir = Path(input_dir)
    output_base = Path(output_base)

    raw_files = list(input_dir.glob("*.raw"))

    for raw_file in raw_files:
        for polarity in ["positive", "negative"]:
            output_dir = output_base / polarity
            output_dir.mkdir(parents=True, exist_ok=True)
            run_msconvert(str(raw_file), output_dir, polarity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split RAW file into positive/negative mzMLs with MSConvert.")
    parser.add_argument("-i", "--input", required=True, help="Input directory with RAW files.")
    parser.add_argument("-o", "--output", required=True, help="Output base directory.")

    args = parser.parse_args()
    convert_all(args.input, args.output)
