"""Fetch a named upstream ref and record the resolved commit. Does not install it.
Only run this with a trusted upstream and an explicitly selected ref.
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--ref', required=True, help='Prefer a full audited commit hash')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'Refusing to overwrite {a.out}')
    if not a.url.startswith('https://github.com/') or a.ref.startswith('-'):
        raise SystemExit('Expected HTTPS GitHub URL and non-option ref')
    a.out.mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(a.out), *args], text=True).strip()
    git('init'); git('remote', 'add', 'origin', a.url)
    git('fetch', '--depth', '1', 'origin', a.ref)
    git('checkout', '--detach', 'FETCH_HEAD')
    lock = {'url': a.url, 'requested_ref': a.ref, 'resolved_commit': git('rev-parse','HEAD')}
    (a.out.parent / (a.out.name + '.lock.json')).write_text(json.dumps(lock,indent=2))
    print(json.dumps(lock,indent=2))

if __name__ == '__main__':
    main()
