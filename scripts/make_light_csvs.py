"""Create column-filtered CSVs for the official loader with bounded memory."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
import yaml


def recording_from_filename(name):
    match=re.fullmatch(r'data-(.+)-\d{4}-of-\d{4}\.csv',name)
    if not match:raise ValueError(f'Unexpected source filename: {name}')
    return match.group(1)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',required=True,type=Path)
    p.add_argument('--configs',required=True,nargs='+',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--chunk-rows',type=int,default=100000);a=p.parse_args()
    if a.chunk_rows<1:raise ValueError('chunk-rows must be positive')
    columns=[];selected=set();config_hashes={}
    for path in a.configs:
        cfg=yaml.safe_load(path.read_text());config_hashes[str(path.resolve())]=digest(path)
        for column in cfg['include_columns']:
            if column not in columns:columns.append(column)
        for split in ['train','val']:
            values=cfg.get(f'include_recordings_{split}',[])
            if values!='all':selected.update(values)
    a.out.mkdir(parents=True,exist_ok=True);report=[]
    for source in sorted(a.raw_root.glob('data-*.csv')):
        recording=recording_from_filename(source.name)
        if recording not in selected:continue
        out=a.out/(source.stem+'_light.csv')
        state={'recording':recording,'source':source.name,'source_size':source.stat().st_size,
               'output':out.name,'columns':columns}
        side=out.with_suffix('.json')
        if out.is_file() and side.is_file():
            old=json.loads(side.read_text())
            if all(old.get(k)==v for k,v in state.items()):
                report.append(old);print(f'SKIP {recording}',flush=True);continue
        if out.exists():out.unlink()
        rows=0
        for i,chunk in enumerate(pd.read_csv(source,usecols=columns,chunksize=a.chunk_rows)):
            chunk.to_csv(out,index=False,mode='a',header=i==0);rows+=len(chunk)
        state.update(rows=rows,output_size=out.stat().st_size,output_sha256=digest(out))
        side.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n');report.append(state)
        print(f'LIGHT {len(report)}/{len(selected)} {recording}: {rows} rows, {out.stat().st_size/1e6:.1f} MB',flush=True)
    missing=sorted(selected-{x['recording'] for x in report})
    iz=a.raw_root/'interaction_zone_center_positions.json'
    if iz.is_file() and not (a.out/iz.name).exists():(a.out/iz.name).symlink_to(iz.resolve())
    provenance={'schema':'acie.light-csv.v1','configs':config_hashes,'selected_count':len(selected),
                'created_count':len(report),'missing_recordings':missing,'files':report,
                'transformation':'pandas chunked column projection; row order and values retained'}
    (a.out/'light_csv_provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'created_count':len(report),'missing_recordings':missing},ensure_ascii=False))


if __name__=='__main__':main()
