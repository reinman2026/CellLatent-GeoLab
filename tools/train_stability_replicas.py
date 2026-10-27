"""Run many independent RegVelo fits and select five geometric representatives.

The existing three-model page remains unchanged. This experiment keeps every
run under .runtime/stability-runs and writes a compact summary plus five full
representative models for a later stability page.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path('E:/PycharmProjects/rna_window_response_hlc_snapshot-regnet/ref/regvelo/localtest/.venv_regvelo/Scripts/python.exe')


def run_one(python: Path, run_index: int, seed: int, args: argparse.Namespace, output: Path) -> dict:
    run_dir = output / f'run-{run_index:02d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir / 'model-1/result.json'
    if args.resume_existing and existing.is_file():
        result = json.loads(existing.read_text(encoding='utf-8'))
        return {'run': run_index, 'seed': seed, 'status': 'complete', 'epochs': result.get('epochs'),
                'bestEpoch': result.get('bestEpoch'), 'path': str(existing), 'resumed': True}
    command = [str(python), str(ROOT / 'tools/train_regvelo_replicas.py'),
               '--models', '1', '--seed', str(seed), '--epochs', str(args.epochs),
               '--samples', str(args.samples), '--hidden', str(args.hidden),
               '--threads', str(args.threads), '--workers', '1', '--no-publish',
               '--output', str(run_dir)]
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'NUMBA_DISABLE_JIT': '1'}
    log_path = run_dir / 'run.log'
    with log_path.open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result_path = run_dir / 'model-1/result.json'
    if completed.returncode or not result_path.is_file():
        return {'run': run_index, 'seed': seed, 'status': 'failed', 'returncode': completed.returncode,
                'error': (run_dir / 'model-1/error.json').read_text(encoding='utf-8') if (run_dir / 'model-1/error.json').is_file() else 'missing result.json'}
    result = json.loads(result_path.read_text(encoding='utf-8'))
    result['stabilityRun'] = run_index
    result['stabilitySeed'] = seed
    result_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return {'run': run_index, 'seed': seed, 'status': 'complete', 'epochs': result.get('epochs'),
            'bestEpoch': result.get('bestEpoch'), 'path': str(result_path)}


def distance_signature(mu: list[list[float]]) -> list[float]:
    values = []
    for i in range(len(mu) - 1):
        for j in range(i + 1, len(mu)):
            values.append(sum((mu[i][k] - mu[j][k]) ** 2 for k in range(3)) ** .5)
    scale = sorted(values)[len(values) // 2] or 1.0
    return [value / scale for value in values]


def squared_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a))


def choose_representatives(results: list[dict], count: int = 5) -> list[int]:
    signatures = [distance_signature(item['mu']) for item in results]
    n = len(results)
    pairwise = [[squared_distance(signatures[i], signatures[j]) for j in range(n)] for i in range(n)]
    medoids = [min(range(n), key=lambda i: sum(pairwise[i]))]
    while len(medoids) < min(count, n):
        candidate = max((i for i in range(n) if i not in medoids), key=lambda i: min(pairwise[i][j] for j in medoids))
        medoids.append(candidate)
    for _ in range(20):
        updated = []
        for cluster, old in enumerate(medoids):
            members = [i for i in range(n) if min(range(len(medoids)), key=lambda k: pairwise[i][medoids[k]]) == cluster]
            updated.append(min(members or [old], key=lambda i: sum(pairwise[i][j] for j in (members or [old]))))
        if updated == medoids:
            break
        medoids = updated
    return sorted(medoids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--base-seed', type=int, default=11)
    parser.add_argument('--seed-step', type=int, default=18)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--output', type=Path, default=ROOT / '.runtime/stability-runs')
    parser.add_argument('--resume-existing', action='store_true', help='Reuse completed run directories.')
    args = parser.parse_args()
    if not 5 <= args.runs <= 100:
        parser.error('--runs must be between 5 and 100')
    python = args.python if args.python.is_file() else Path(sys.executable)
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    seeds = [args.base_seed + i * args.seed_step for i in range(args.runs)]
    manifest = {'status': 'running', 'startedAt': datetime.now(timezone.utc).isoformat(), 'runs': args.runs, 'seeds': seeds,
                'config': {'epochs': args.epochs, 'samples': args.samples, 'hidden': args.hidden, 'threads': args.threads}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    summaries = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, args.runs))) as pool:
        futures = {pool.submit(run_one, python, i + 1, seed, args, output): i for i, seed in enumerate(seeds)}
        for future in as_completed(futures):
            try:
                item = future.result();summaries.append(item);print(json.dumps(item, ensure_ascii=False), flush=True)
            except Exception:
                item = {'run': futures[future] + 1, 'seed': seeds[futures[future]], 'status': 'failed', 'error': traceback.format_exc()}
                summaries.append(item);print(json.dumps(item, ensure_ascii=False), flush=True)
    complete = []
    for summary in summaries:
        if summary.get('status') != 'complete':
            continue
        result = output / f"run-{summary['run']:02d}/model-1/result.json"
        if result.is_file():
            complete.append(json.loads(result.read_text(encoding='utf-8')))
    complete.sort(key=lambda item: item.get('stabilityRun', 0))
    if len(complete) < 5:
        manifest.update({'status': 'failed', 'summaries': sorted(summaries, key=lambda x: x['run']), 'complete': len(complete)})
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return 2
    representative_positions = choose_representatives(complete, 5)
    representatives = [complete[i] for i in representative_positions]
    bundle = {'format': 'regvelo-stability-v1', 'status': 'complete', 'generatedAt': datetime.now(timezone.utc).isoformat(),
              'representativePositions': representative_positions,
              'representatives': representatives,
              'summaries': sorted(summaries, key=lambda x: x['run']),
              'provenance': {'training': 'independent local RegVelo fits', 'selection': 'five geometry medoids from normalized pairwise-distance signatures',
                             'data': 'same website 24-cell, 3-gene observations; this experiment tests seed stability'}}
    (output / 'stability.json').write_text(json.dumps(bundle, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    target = ROOT / 'dist/stability-results.mjs'
    target.write_text('export const stabilityResults = ' + json.dumps(bundle, ensure_ascii=False, allow_nan=False) + ';\n', encoding='utf-8')
    manifest.update({'status': 'complete', 'finishedAt': datetime.now(timezone.utc).isoformat(), 'summaries': sorted(summaries, key=lambda x: x['run']),
                     'complete': len(complete), 'representativePositions': representative_positions})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'complete': len(complete), 'representativeRuns': [representatives[i]['stabilityRun'] for i in range(len(representatives))]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
