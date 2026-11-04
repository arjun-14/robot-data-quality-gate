"""Download the metadata + parquet files of the demo dataset (no video; ~a few MB)."""
import sys
from huggingface_hub import snapshot_download

REPO = "lerobot/svla_so100_pickplace"
dest = sys.argv[1] if len(sys.argv) > 1 else "data/so100"
snapshot_download(REPO, repo_type="dataset", local_dir=dest, allow_patterns=["meta/*", "meta/**", "data/**"])
print("saved to", dest)
