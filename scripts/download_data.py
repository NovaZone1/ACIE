"""Explicit processed-data download; raw videos are not requested."""
import argparse
from pathlib import Path
from acie.io import write_json
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--revision',required=True)
p.add_argument('--accept-data-terms',action='store_true');a=p.parse_args()
if not a.accept_data_terms:raise SystemExit('Review the original dataset terms, then pass --accept-data-terms')
from huggingface_hub import HfApi,snapshot_download
info=HfApi().dataset_info('rlorlou/HUI360',revision=a.revision)
path=snapshot_download('rlorlou/HUI360',repo_type='dataset',revision=info.sha,local_dir=a.out,
                       allow_patterns=['*.csv','*.json','README.md','LICENSE*'])
write_json(Path(a.out)/'acie_download_lock.json',{'repo':'rlorlou/HUI360','requested':a.revision,'resolved':info.sha})
print(path)
