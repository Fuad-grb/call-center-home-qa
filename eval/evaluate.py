"""
Evaluation script — measures pipeline accuracy against eval dataset.

Usage:
    python eval/evaluate.py                          # full dataset
    python eval/evaluate.py --limit 5                # first 5 only
    python eval/evaluate.py --output eval/results.json
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.logger import setup_logging, get_logger
from app.pipeline import evaluate_call

setup_logging()
logger = get_logger(__name__)

CRITERIA = ["KR2.1", "KR2.2", "KR2.3", "KR2.4", "KR2.5"]


def run_evaluation(eval_path: str, output_path: str | None = None, limit: int | None = None):
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if limit:
        eval_data = eval_data[:limit]

    print(f"\nEvaluating {len(eval_data)} items...\n")

    exact_match = defaultdict(int)
    within_one = defaultdict(int)
    total = defaultdict(int)
    detailed = []

    for i, item in enumerate(eval_data):
        dataset_id = item.get("dataset_id", f"item_{i}")
        call_input = item["input"]
        expected = item["expected_output"]

        print(f"  [{i+1}/{len(eval_data)}] {dataset_id}...", end=" ", flush=True)

        try:
            result = evaluate_call(call_input)
        except Exception as e:
            print(f"ERROR: {e}")
            detailed.append({"dataset_id": dataset_id, "error": str(e)})
            continue

        item_detail = {"dataset_id": dataset_id, "criteria": {}}

        for cr in CRITERIA:
            exp_score = expected.get(cr, {}).get("score")
            pred = result.scores.get(cr)
            if exp_score is None or pred is None:
                continue

            total[cr] += 1
            is_exact = pred.score == exp_score
            is_close = abs(pred.score - exp_score) <= 1

            if is_exact:
                exact_match[cr] += 1
            if is_close:
                within_one[cr] += 1

            item_detail["criteria"][cr] = {
                "expected": exp_score,
                "predicted": pred.score,
                "exact": is_exact,
                "reasoning": pred.reasoning,
            }

        detailed.append(item_detail)
        # Show quick result
        misses = [cr for cr in CRITERIA
                  if item_detail["criteria"].get(cr, {}).get("exact") is False]
        if misses:
            print(f"MISSES: {misses}")
        else:
            print("OK")

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    total_exact = total_within = total_count = 0

    for cr in CRITERIA:
        n = total[cr]
        if n == 0:
            continue
        ex_pct = exact_match[cr] / n * 100
        w1_pct = within_one[cr] / n * 100
        total_exact += exact_match[cr]
        total_within += within_one[cr]
        total_count += n
        print(f"  {cr}: exact={exact_match[cr]}/{n} ({ex_pct:.1f}%)  ±1={within_one[cr]}/{n} ({w1_pct:.1f}%)")

    if total_count > 0:
        print(f"\n  OVERALL: exact={total_exact}/{total_count} ({total_exact/total_count*100:.1f}%)  ±1={total_within}/{total_count} ({total_within/total_count*100:.1f}%)")

    print("=" * 60)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detailed, ensure_ascii=False, indent=2, fp=f)
        print(f"\nDetails saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="eval/eval_dataset.json")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--limit", "-n", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(args.eval, args.output, args.limit)
