"""Stage an auditable offline subset for the official HUI360 loader.

Real selected CSVs are symlinked. Unselected official files receive one-row light
placeholders because the upstream offline loader checks all 97 filenames before
filtering by recording. Placeholders can never satisfy a selected recording.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import yaml


def recording_from_filename(name):
    stem=re.sub(r'_light(?=\.csv$)','',name)
    match=re.fullmatch(r'data-(.+)-\d{4}-of-\d{4}\.csv',stem)
    if not match:raise ValueError(f'Unexpected official filename: {name}')
    return match.group(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',required=True,type=Path);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--split',choices=['train','val'],required=True)
    p.add_argument('--allow-missing-recording',action='append',default=[])
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError('Staging directory is not empty')
    a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.upstream.resolve()))
    from datasets.HUIDatasetUtils import MAIN_DATASET_FILENAMES
    cfg=yaml.safe_load(a.config.read_text());selected=cfg[f'include_recordings_{a.split}']
    if selected=='all':raise ValueError('Subset staging requires an explicit recording list')
    by_recording={recording_from_filename(name):name for name in MAIN_DATASET_FILENAMES}
    absent=sorted(set(selected)-set(by_recording))
    unauthorized=sorted(set(absent)-set(a.allow_missing_recording))
    if unauthorized:raise FileNotFoundError(f'Selected recordings absent from official filename map: {unauthorized}')
    present=[];placeholders=[];missing_real=[]
    for name in MAIN_DATASET_FILENAMES:
        source=a.data_root/name
        light_name=Path(name).with_suffix('').name+'_light.csv'
        light_source=a.data_root/light_name
        recording=recording_from_filename(name)
        if light_source.is_file():
            (a.out/light_name).symlink_to(light_source.resolve());present.append(light_name)
        elif source.is_file():
            (a.out/name).symlink_to(source.resolve());present.append(name)
        elif recording in selected:
            missing_real.append(recording)
        else:
            (a.out/light_name).write_text(f'recording\n{recording}\n')
            placeholders.append(light_name)
    if missing_real:raise FileNotFoundError(f'Selected recordings lack real CSVs: {sorted(missing_real)}')
    iz=a.data_root/'interaction_zone_center_positions.json'
    if cfg.get('do_recenter_interaction_zone'):
        if not iz.is_file():raise FileNotFoundError(f'Missing {iz}')
        (a.out/iz.name).symlink_to(iz.resolve())
    report={'schema':'acie.source-staging.v1','config':str(a.config.resolve()),'split':a.split,
            'selected_count':len(selected),'selected_absent_from_frozen_revision':absent,
            'allowed_missing_recordings':sorted(a.allow_missing_recording),'real_csv_count':len(present),
            'placeholder_count':len(placeholders),'real_csvs':present,'placeholders':placeholders}
    (a.out/'staging_provenance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},indent=2))


if __name__=='__main__':main()
