"""Print installed versions and compute availability without downloads."""
import importlib.metadata
import json
import platform
import sys
names=['numpy','torch','scikit-learn','PyYAML','pytest','pandas','opencv-python-headless','transformers','huggingface-hub']
versions={}
for name in names:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
result={'python':sys.version,'platform':platform.platform(),'packages':versions}
try:
    import torch
    result['cuda_available']=torch.cuda.is_available();result['torch_cuda_version']=torch.version.cuda
except ImportError:result['cuda_available']=False
print(json.dumps(result,indent=2))
