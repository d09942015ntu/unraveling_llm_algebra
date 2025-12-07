import pandas as pd
import argparse
import os

def parquet_to_csv(input_path, output_path=None, chunksize=None):
    """
    Convert a parquet file to CSV.

    Args:
        input_path (str): Path to the .parquet file.
        output_path (str): Path to the output .csv file.
        chunksize (int): Optional. Number of rows per chunk to write (for large files).
    """
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".csv"

    print(f"Reading parquet: {input_path}")

    # Standard read
    if chunksize is None:
        df = pd.read_parquet(input_path)
        df.to_csv(output_path, index=False)
        print(f"Saved CSV: {output_path}")
        return

    # Chunked write for huge parquet files
    df_iter = pd.read_parquet(input_path, chunksize=chunksize)
    first = True

    for chunk in df_iter:
        chunk.to_csv(output_path, mode="w" if first else "a",
                     index=False, header=first)
        first = False

    print(f"Saved CSV (chunked): {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .parquet to .csv")
    parser.add_argument("--input", help="Input .parquet file path", default="./data/MAWPS/test-00000-of-00001.parquet")
    parser.add_argument("--output", "-o", help="Optional output .csv path", default="./data/MAWPS/test-00000-of-00001.csv")
    parser.add_argument("--chunksize", "-c", type=int,
                        help="Optional chunk size for large files")

    args = parser.parse_args()
    parquet_to_csv(args.input, args.output, args.chunksize)



