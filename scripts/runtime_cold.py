"""Fresh-process import/load/first-prediction and process high-water RSS."""

import argparse
import json
import resource
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runtime")
    p.add_argument("threads", type=int)
    args = p.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "artifacts/i1/client/catalog.json").read_text())
    case = json.loads((root / "artifacts/i1/client" / catalog["cases"][0]["asset"]).read_text())
    scaling = catalog["models"]["timexer"]["scaling"]
    import numpy as np

    h = np.asarray([case["history"]], dtype=np.float32)
    h[:, :, :3] = (h[:, :, :3] - scaling["means"]) / scaling["scales"]
    c = np.asarray([case["target_calendar"]], dtype=np.float32)
    if args.runtime in ("eager", "torchscript"):
        import torch

        torch.set_num_threads(args.threads)
        if args.runtime == "eager":
            from gridlens.models.bundle import NeuralBundle

            baseline = root / "reports/local/i0-public"
            report = json.loads((baseline / "baseline_reproduction.json").read_text())
            row = next(r for r in report["results"] if r["kind"] == "timexer")
            model = NeuralBundle.load_bundle(baseline / row["bundle"]).model
        else:
            model = torch.jit.load(str(root / "artifacts/release/runtime/timexer.ts"))
        with torch.no_grad():
            values = model(torch.from_numpy(h), torch.from_numpy(c)).numpy()
    else:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        file = "timexer-int8.onnx" if args.runtime == "onnx_int8" else "timexer.onnx"
        session = ort.InferenceSession(
            str(root / "artifacts/release/runtime" / file),
            options,
            providers=["CPUExecutionProvider"],
        )
        values = session.run(["prediction"], {"history": h, "calendar": c})[0]
    print(
        json.dumps(
            {
                "runtime": args.runtime,
                "threads": args.threads,
                "child_import_load_first_predict_ms": (time.perf_counter() - started) * 1000,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "normalised_prediction": values.tolist(),
                "rss_scope": "Linux fresh interpreter; runtime-specific imports, verified model, one input and first prediction",
            }
        )
    )


if __name__ == "__main__":
    main()
