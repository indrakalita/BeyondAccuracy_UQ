
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files

snapshot_download(repo_id="blanchon/ETCI-2021-Flood-Detection",
                  repo_type="dataset",
                  allow_patterns="data/train/*",  # This is the magic line to skip everything else
                  local_dir="./ETCI_dataset/New/",
                  token="<><><>",
                  resume_download=True,           # Explicitly tell it to resume
                  max_workers=1                  # Don't overwhelm the API (prevents 429)
                 )
