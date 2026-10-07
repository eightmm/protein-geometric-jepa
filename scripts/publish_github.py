#!/usr/bin/env python3
"""Create a NEW private GitHub repository using the user's authenticated gh CLI.

Never requests a token, changes global git configuration, overwrites an existing
repository, or pushes with --force. This script is not run by artifact creation.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


def run(args, cwd, capture=False):
    return subprocess.run(args, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def publish(repo: str, root: Path, dry_run=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9._-]+', repo):
        raise ValueError('Expected owner/repository, not a URL or filesystem path.')
    if not (root/'pyproject.toml').is_file() or not (root/'src/protein_jepa').is_dir():
        raise ValueError('Run this script from the distributed project, not another repository.')
    if dry_run:
        print(f'Would verify gh authentication and owner, initialize local Git if necessary, '
              f'and create NEW PRIVATE repository {repo}. No commands executed.')
        return
    for exe in ('git', 'gh'):
        if shutil.which(exe) is None:
            raise RuntimeError(f'{exe} is not installed. Install it and authenticate with gh auth login locally.')
    run(['gh', 'auth', 'status'], root)
    login = run(['gh', 'api', 'user', '--jq', '.login'], root, True).stdout.strip()
    if repo.split('/')[0].lower() != login.lower():
        raise ValueError(f'This safe publisher only creates repositories in your authenticated account ({login}).')
    # Listing your owned repositories avoids interpreting an authentication/
    # connection error or a transient 404 as "repository does not exist".
    owned = run(['gh', 'api', '--paginate', 'user/repos?affiliation=owner&per_page=100',
                 '--jq', '.[].full_name'], root, True).stdout.splitlines()
    if repo.lower() in {r.lower() for r in owned}:
        raise FileExistsError(f'{repo} already exists. Refusing to repurpose or overwrite it.')
    if not (root/'.git').exists():
        run(['git', 'init', '-b', 'main'], root)
    else:
        top = Path(run(['git', 'rev-parse', '--show-toplevel'], root, True).stdout.strip()).resolve()
        if top != root.resolve():
            raise RuntimeError('Refusing to operate in a parent Git repository.')
    if run(['git', 'remote'], root, True).stdout.strip():
        raise RuntimeError('A Git remote already exists. Inspect it manually; this script will not change it.')
    status = run(['git', 'status', '--porcelain'], root, True).stdout.strip()
    head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=root,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if status or head.returncode:
        print('Files to commit (ignored datasets, checkpoints and .env are excluded):')
        print(status)
        run(['git', 'add', '--all'], root)
        run(['git', '-c', f'user.name={login}', '-c', f'user.email={login}@users.noreply.github.com',
             'commit', '-m', 'Initialize protein geometric JEPA implementation and specifications'], root)
    run(['gh', 'repo', 'create', repo, '--private', '--source', str(root), '--remote', 'origin', '--push'], root)
    print(f'Created and pushed private repository: https://github.com/{repo}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='eightmm/protein-geometric-jepa')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    try:
        publish(args.repo, Path(__file__).resolve().parents[1], args.dry_run)
    except (ValueError, RuntimeError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f'Publication stopped: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
