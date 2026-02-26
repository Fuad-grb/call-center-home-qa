"""
CLI entry point.

Usage:
    python main.py --input transcript.json
    python main.py --input eval_dataset.json --batch
    python main.py --input eval_dataset.json --batch --output results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from app.logger import setup_logging, get_logger
from app.pipeline import evaluate_call

setup_logging()
logger = get_logger(__name__)


def evaluate_single(input_data: dict) -> dict:
    result = evaluate_call(input_data)
    return result.model_dump()


def evaluate_batch(items: list[dict]) -> list[dict]:
    results = []
    for i, item in enumerate(items):
        call_input = item.get("input", item)
        logger.info(f"Processing {i + 1}/{len(items)}", extra={
            "call_id": call_input.get("call_id", "?"),
            "dataset_id": item.get("dataset_id", "?"),
        })
        result = evaluate_single(call_input)
        results.append({
            "dataset_id": item.get("dataset_id", f"item_{i}"),
            "call_id": call_input.get("call_id", "?"),
            "evaluation": result,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Kontakt Home QA Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--batch", action="store_true", help="Batch mode for eval dataset")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.batch:
        if not isinstance(data, list):
            logger.error("Batch mode expects a JSON array")
            sys.exit(1)
        results = evaluate_batch(data)
    else:
        results = evaluate_single(data)

    output_json = json.dumps(results, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        logger.info(f"Saved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
