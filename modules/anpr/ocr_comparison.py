"""Compare both recognizers on the saved shortlist from an earlier plate scan."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from .ocr_candidates import candidate_text, engine_report
from .ocr_session import OCRSession


def compare(manifest, output, project_root):
    prior = json.loads(manifest.read_text(encoding="utf-8"))
    observations = [row for row in prior["plates"] if row["ocr_inputs"]]
    if not observations:
        raise ValueError("The supplied manifest has no OCR shortlist")
    session = OCRSession({"ocr_engine": "ppocrv4", "compare_ocr_engines": True}, project_root)
    rows = []
    repeats = defaultdict(lambda: defaultdict(set))
    variants = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        image = cv2.imdecode(np.frombuffer(Path(observation["crop_path"]).read_bytes(), np.uint8), 1)
        if image is None:
            raise OSError(f"Could not decode {observation['crop_path']}")
        _, engines = session.scan(image)
        track_id, frame = observation["track_id"], observation["frame"]
        for name, engine in engines.items():
            result = engine_report(engine)
            best = result["best"] or {}
            for read in result["readings"]:
                variants[name][read["variant"]].append(read)
                text = candidate_text(read["raw_text"])
                if text and read["confidence"] >= 0.45:
                    repeats[name][(track_id, text)].add(frame)
            rows.append({
                "track_id": track_id, "frame": frame, "width": image.shape[1], "height": image.shape[0],
                "engine": name, "raw_text": best.get("raw_text", ""), "confidence": best.get("confidence"),
                "elapsed_ms": round(engine.elapsed_ms, 3), "indian_format_valid": best.get("format_valid", False),
                "error": engine.error, "variants": result["readings"],
            })
    summary = {}
    for name in ("easyocr", "ppocrv4"):
        engine_rows = [row for row in rows if row["engine"] == name]
        extracted = [row for row in engine_rows if row["raw_text"]]
        repeated = [
            {"track_id": tid, "text": text, "frames": sorted(frames)}
            for (tid, text), frames in repeats[name].items() if len(frames) >= 2
        ]
        durations = [row["elapsed_ms"] for row in engine_rows if row["elapsed_ms"] > 0]
        summary[name] = {
            "observations": len(engine_rows), "nonempty_text": len(extracted),
            "failed_or_empty": len(engine_rows) - len(extracted),
            "engine_errors": sum(bool(row["error"]) for row in engine_rows),
            "average_confidence_nonempty": mean(row["confidence"] for row in extracted) if extracted else None,
            "average_processing_ms": mean(durations) if durations else None,
            "repeated_candidates": repeated,
            "tracks_with_repeated_candidates": len({entry["track_id"] for entry in repeated}),
            "variants": {
                variant: {
                    "observations": len(reads),
                    "nonempty_outputs": sum(bool(read["raw_text"].strip()) for read in reads),
                    "format_matches": sum(read["format_valid"] for read in reads),
                    "candidate_outputs_at_045": sum(
                        bool(candidate_text(read["raw_text"])) and read["confidence"] >= 0.45 for read in reads
                    ),
                }
                for variant, reads in variants[name].items()
            },
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ocr-comparison.json").write_text(json.dumps({
        "observations": len(observations), "engines": summary,
        "initialization_errors": session.initialization_errors, "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [key for key in rows[0] if key != "variants"]
    with (output / "ocr-comparison.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            raw = row["raw_text"]
            writer.writerow({**row, "raw_text": "'" + raw if raw.lstrip().startswith(("=", "+", "-", "@")) else raw})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    print(json.dumps(compare(arguments.manifest, arguments.output, Path(__file__).resolve().parents[2]), indent=2))
