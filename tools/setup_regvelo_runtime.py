"""Install the complete dependencies in this workspace's isolated venv."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    target = ROOT / '.runtime' / 'regvelo-venv'
    python = target / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    command = [str(python), '-m', 'pip', 'install', '--index-url',
               'https://pypi.org/simple', '--cache-dir', str(ROOT / '.runtime' / 'pip-cache'),
               '-r', str(ROOT / 'tools/requirements-regvelo.txt')]
    if args.plan:
        print('Isolated environment:', target)
        print(subprocess.list2cmdline(command))
        return 0
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(target)
    return subprocess.call(command, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())

