"""Run twenty independent RegVelo fits on one six-gene paper toy dataset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("E:/PycharmProjects/rna_window_response_hlc_snapshot-regnet/ref/regvelo/localtest/.venv_regvelo/Scripts/python.exe")
DEFAULT_SOURCE = Path("E:/PycharmProjects/rna_window_response_hlc_snapshot-regnet/ref/regvelo")
DEFAULT_OBSERVATIONS = ROOT / ".runtime/paper-toy/observations.json"
DEFAULT_OUTPUT = ROOT / ".runtime/paper-toy/stability"
DEFAULT_BUNDLE = ROOT / "dist/paper-toy-stability-results.mjs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(python: Path, run_index: int, seed: int, args: argparse.Namespace, output: Path) -> dict:
    run_dir = output / f"run-{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "model-1/result.json"
    if args.resume_existing and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return {"run": run_index, "seed": seed, "status": "complete", "epochs": result.get("epochs"),
                "bestEpoch": result.get("bestEpoch"), "path": str(result_path), "resumed": True}
    command = [str(python), str(ROOT / "tools/train_regvelo_replicas.py"), "--models", "1",
               "--seed", str(seed), "--source", str(args.source), "--observations", str(args.observations),
               "--epochs", str(args.epochs), "--samples", str(args.samples), "--hidden", str(args.hidden),
               "--latent", "3", "--threads", str(args.threads), "--workers", "1", "--no-publish",
               "--output", str(run_dir)]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "NUMBA_DISABLE_JIT": "1"}
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode or not result_path.is_file():
        error_path = run_dir / "model-1/error.json"
        return {"run": run_index, "seed": seed, "status": "failed", "returncode": completed.returncode,
                "error": error_path.read_text(encoding="utf-8") if error_path.is_file() else "missing result.json"}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update({"stabilityRun": run_index, "stabilitySeed": seed})
    result_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return {"run": run_index, "seed": seed, "status": "complete", "epochs": result.get("epochs"),
            "bestEpoch": result.get("bestEpoch"), "path": str(result_path)}


def distance_signature(mu: list[list[float]]) -> list[float]:
    values = [sum((mu[i][k] - mu[j][k]) ** 2 for k in range(3)) ** 0.5
              for i in range(len(mu) - 1) for j in range(i + 1, len(mu))]
    scale = sorted(values)[len(values) // 2] or 1.0
    return [value / scale for value in values]


def choose_representatives(results: list[dict], count: int = 5) -> list[int]:
    signatures = [distance_signature(item["mu"]) for item in results]
    distance = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a))
    pairwise = [[distance(signatures[i], signatures[j]) for j in range(len(results))]
                for i in range(len(results))]
    medoids = [min(range(len(results)), key=lambda i: sum(pairwise[i]))]
    while len(medoids) < min(count, len(results)):
        medoids.append(max((i for i in range(len(results)) if i not in medoids),
                           key=lambda i: min(pairwise[i][j] for j in medoids)))
    for _ in range(20):
        updated = []
        for cluster, old in enumerate(medoids):
            members = [i for i in range(len(results))
                       if min(range(len(medoids)), key=lambda k: pairwise[i][medoids[k]]) == cluster]
            updated.append(min(members or [old], key=lambda i: sum(pairwise[i][j] for j in (members or [old]))))
        if updated == medoids:
            break
        medoids = updated
    return sorted(medoids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=11)
    parser.add_argument("--seed-step", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    if not 5 <= args.runs <= 100:
        parser.error("--runs must be between 5 and 100")
    if not args.observations.is_file():
        parser.error(f"Missing observations: {args.observations}")
    if not args.source.is_dir():
        parser.error(f"Missing local RegVelo source: {args.source}")
    python = args.python if args.python.is_file() else Path(sys.executable)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = [args.base_seed + i * args.seed_step for i in range(args.runs)]
    manifest = {"status": "running", "startedAt": datetime.now(timezone.utc).isoformat(),
                "runs": args.runs, "seeds": seeds,
                "config": {"epochs": args.epochs, "samples": args.samples, "hidden": args.hidden,
                           "threads": args.threads, "latent": 3,
                           "observations": str(args.observations.resolve()),
                           "observationSha256": digest(args.observations)}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, args.runs))) as pool:
        futures = {pool.submit(run_one, python, i + 1, seed, args, output): i
                   for i, seed in enumerate(seeds)}
        for future in as_completed(futures):
            try:
                item = future.result()
            except Exception:
                index = futures[future] + 1
                item = {"run": index, "seed": seeds[index - 1], "status": "failed",
                        "error": traceback.format_exc()}
            summaries.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    complete = []
    for summary in summaries:
        if summary.get("status") != "complete":
            continue
        result = output / f"run-{summary['run']:02d}/model-1/result.json"
        if result.is_file():
            complete.append(json.loads(result.read_text(encoding="utf-8")))
    complete.sort(key=lambda item: item.get("stabilityRun", 0))
    summaries.sort(key=lambda item: item["run"])
    if len(complete) < args.runs:
        manifest.update({"status": "failed", "summaries": summaries, "complete": len(complete)})
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2
    representative_positions = choose_representatives(complete, 5)
    data = json.loads(args.observations.read_text(encoding="utf-8"))
    bundle = {"format": "regvelo-paper-toy-stability-v1", "status": "complete",
              "generatedAt": datetime.now(timezone.utc).isoformat(), "expectedModelCount": args.runs,
              "representativePositions": representative_positions,
              "representatives": [complete[i] for i in representative_positions],
              "models": complete, "data": data, "summaries": summaries,
              "provenance": {"training": "independent complete local RegVelo fits",
                             "data": "one shared six-gene Hill toy dataset", "latent": "three dimensions for every fit",
                             "selection": "five geometry medoids from normalized pairwise-distance signatures",
                             "observationSha256": digest(args.observations)}}
    (output / "stability.json").write_text(json.dumps(bundle, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text("export const paperToyStability = " + json.dumps(bundle, ensure_ascii=False, allow_nan=False) + ";\n",
                           encoding="utf-8")
    manifest.update({"status": "complete", "finishedAt": datetime.now(timezone.utc).isoformat(),
                     "summaries": summaries, "complete": len(complete),
                     "representativePositions": representative_positions})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "complete": len(complete),
                      "representativeRuns": [complete[i]["stabilityRun"] for i in representative_positions],
                      "bundle": str(args.bundle)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
