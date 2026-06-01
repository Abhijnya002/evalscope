"""Re-build the slim calibration data under evalscope_ext/calibration/data/.

Reads the full LFS-backed reviews + predictions from the Cerebras
challenge repo (`Evals/Part 1/`, `Evals/MMMU/`) and writes a shrunken
copy with only the fields the pruners actually consume. The full data
is ~920 MB; the slim copy is ~200 KB.

Usage:

    python scripts/build_slim_calibration.py \\
        --src /path/to/challenge/Evals \\
        --dst evalscope_ext/calibration/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LCB_MODELS = ["gpt-oss-120b", "kimi-k2.5", "minimax-m2.5"]
AA_LCR_MODELS = LCB_MODELS  # same three

LCB_METADATA_KEYS = ("question_id", "difficulty", "platform", "contest_date", "tags")
AA_LCR_METADATA_KEYS = ("num_documents", "context_tokens", "total_tokens", "category")


def _slim_lcb(src: Path, dst: Path) -> None:
    out_dir = dst / "live_code_bench_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in LCB_MODELS:
        reviews: dict[int, float] = {}
        for line in (src / "Part 1" / "reviews" / f"live_code_bench_v5__{model}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            reviews[r["index"]] = float(r["sample_score"]["score"]["value"]["pass"])

        preds_meta: dict[int, dict] = {}
        for line in (src / "Part 1" / "predictions" / f"live_code_bench_v5__{model}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            md = p.get("metadata") or {}
            preds_meta[p["index"]] = {k: md.get(k) for k in LCB_METADATA_KEYS}

        lines = [
            json.dumps(
                {
                    "index": idx,
                    "model": model,
                    "score": reviews[idx],
                    "metadata": preds_meta.get(idx, {}),
                }
            )
            for idx in sorted(reviews)
        ]
        (out_dir / f"{model}.jsonl").write_text("\n".join(lines) + "\n")


def _slim_aalcr(src: Path, dst: Path) -> None:
    out_dir = dst / "aa_lcr"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in AA_LCR_MODELS:
        reviews: dict[int, dict] = {}
        for line in (src / "Part 1" / "reviews" / f"aa_lcr__{model}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            score_obj = r["sample_score"]["score"]
            ext = score_obj.get("extracted_prediction", "")
            reviews[r["index"]] = {
                "index": r["index"],
                "model": model,
                "score": float(score_obj["value"]["acc"]),
                "prediction_len": len(ext) if isinstance(ext, str) else 0,
            }

        preds_meta: dict[int, dict] = {}
        for line in (src / "Part 1" / "predictions" / f"aa_lcr__{model}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            md = p.get("metadata") or {}
            preds_meta[p["index"]] = {k: md.get(k) for k in AA_LCR_METADATA_KEYS if md.get(k) is not None}

        lines = []
        for idx in sorted(reviews):
            rec = reviews[idx]
            rec["metadata"] = preds_meta.get(idx, {})
            lines.append(json.dumps(rec))
        (out_dir / f"{model}.jsonl").write_text("\n".join(lines) + "\n")


def _summarize_mmmu(src: Path, dst: Path) -> None:
    out_dir = dst / "mmmu"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    review_root = src / "MMMU" / "reviews" / "glm-4.5v-fp8"
    if not review_root.exists():
        return
    for sub_file in sorted(review_root.iterdir()):
        if not sub_file.name.endswith(".jsonl"):
            continue
        subject = sub_file.stem.replace("mmmu_", "")
        accs = []
        for line in sub_file.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("sample_score", {}).get("score", {}).get("value", {})
            if "acc" in v:
                accs.append(float(v["acc"]))
        summary.append(
            {"subject": subject, "n": len(accs), "mean_acc": sum(accs) / len(accs) if accs else None}
        )
    (out_dir / "glm-4.5v-fp8_per_subject_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Path to challenge Evals/ directory")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    _slim_lcb(args.src, args.dst)
    _slim_aalcr(args.src, args.dst)
    _summarize_mmmu(args.src, args.dst)
    print("Done.")


if __name__ == "__main__":
    main()
