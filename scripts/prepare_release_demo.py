"""Export the frozen selected TimeXer and ridge for fresh public inference."""

import json

from gridlens.data.store import write_once
from scripts.prepare_live_demo import prepare_demo
from scripts.reproduce_baseline import ROOT


def main():
    study = json.loads((ROOT / "artifacts/release/study.json").read_text())
    freeze = json.loads((ROOT / "artifacts/release/point-model-freeze.json").read_text())
    ids = [freeze["choices"]["timexer"], freeze["choices"]["ridge"]]
    selected = [row for row in study["rows"] if row["model_id"] in ids]
    if len(selected) != 2 or freeze["champion_model_id"] != freeze["choices"]["timexer"]:
        raise ValueError("This export expects the frozen selected champion to be TimeXer")
    wrapper = {
        "report_id": study["report_id"],
        "dataset_manifest_id": study["dataset_manifest_id"],
        "results": [
            {"kind": r["kind"], "model_id": r["model_id"], "bundle": str(ROOT / r["bundle"])}
            for r in selected
        ],
    }
    folder = ROOT / "artifacts/release/live-export-input"
    folder.mkdir(parents=True, exist_ok=True)
    write_once(
        folder / "baseline_reproduction.json", (json.dumps(wrapper, indent=2) + "\n").encode()
    )
    output = ROOT / "artifacts/release/live-demo"
    result = prepare_demo(folder, output, ROOT / "data/neso")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
