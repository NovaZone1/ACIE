"""Execute original upstream training unchanged, separate from ACIE's grouped protocol."""
import argparse
import subprocess
import sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--config',required=True)
p.add_argument('extra',nargs=argparse.REMAINDER);a=p.parse_args()
root=Path(a.upstream).resolve();config=Path(a.config).resolve()
cmd=[sys.executable,str(root/'training.py'),'-hp',str(config),'--save_model']
cmd+=a.extra[1:] if a.extra and a.extra[0]=='--' else a.extra
subprocess.run(cmd,cwd=root,check=True)
