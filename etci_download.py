##huggingface-cli download blanchon/ETCI-2021-Flood-Detection --repo-type dataset --local-dir ./my_dataset
#huggingface-cli login --token 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
#huggingface-cli download blanchon/ETCI-2021-Flood-Detection \
#huggingface-cli download blanchon/ETCI-2021-Flood-Detection --include "test/*" --repo-type dataset --local-dir ./ETCI_dataset/New1/ --max-workers 1

from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
'''
repo_id = "blanchon/ETCI-2021-Flood-Detection"
token = "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"
try:
    all_files = list_repo_files(repo_id, repo_type="dataset", token=token)
    # Filter for only the test metadata to start small
    test_files = [f for f in all_files if f.startswith("data/test/")]
    print(f"Success! Found {len(test_files)} files in the test folder.")
except Exception as e:
    print(f"Still blocked by rate limit. Error: {e}")

# 2. If it works, download just the core test metadata file first
# This confirms the connection is working
file_path = hf_hub_download(
    repo_id=repo_id,
    filename="test_metadata.csv", # A single small file
    repo_type="dataset",
    local_dir="./ETCI_dataset/New1/",
    token=token
)
print(f"Confirmed: Small file downloaded to {file_path}")
'''

snapshot_download(repo_id="blanchon/ETCI-2021-Flood-Detection",
                  repo_type="dataset",
                  allow_patterns="data/train/*",  # This is the magic line to skip everything else
                  local_dir="./ETCI_dataset/New/",
                  token="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",
                  resume_download=True,           # Explicitly tell it to resume
                  max_workers=1                  # Don't overwhelm the API (prevents 429)
                 )
