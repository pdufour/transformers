from huggingface_hub import snapshot_download
import os

path = snapshot_download(repo_id='google/timesfm-2.5-200m-flax')
print('Downloaded to', path)
for root, dirs, files in os.walk(path):
    for f in files:
        print(os.path.join(root, f))