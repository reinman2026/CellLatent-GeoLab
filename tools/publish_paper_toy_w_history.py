"""Publish one deterministic paper-toy W trajectory for the teaching page.

The trajectory is copied from a completed local RegVelo fit.  The browser
only replays these recorded states; it never starts a second optimization.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / ".runtime/paper-toy/w-history/run-01/model-1/result.json"
DEFAULT_OBSERVATIONS = ROOT / ".runtime/paper-toy/observations.json"
DEFAULT_OUTPUT = ROOT / "dist/paper-toy-w-history.mjs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    snapshots = result.get("wSnapshots", [])
    if not snapshots:
        raise SystemExit("result has no wSnapshots; rerun train_regvelo_replicas.py")
    genes = observations["genes"]
    prior = observations["prior"]
    if len(genes) != 6 or len(prior["skeleton"]) != 6:
        raise SystemExit("paper toy W history expects the six-gene observations")
    payload = {
        "format": "regvelo-paper-toy-w-history-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "genes": genes,
        "featureOrder": observations.get("featureOrder", []),
        "prior": prior,
        "snapshots": snapshots,
        "snapshotEvery": result.get("snapshotEvery", 10),
        "history": result.get("history", []),
        "model": {
            "index": result.get("index"),
            "seed": result.get("seed"),
            "epochs": result.get("epochs"),
            "bestEpoch": result.get("bestEpoch"),
            "rates": result.get("rates", {}),
            "observationSha256": result.get("observationSha256"),
            "config": result.get("config", {}),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "export const paperToyWHistory = " + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ) + ";\n"
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "snapshots": len(snapshots),
                      "epochs": result.get("epochs"), "seed": result.get("seed")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
