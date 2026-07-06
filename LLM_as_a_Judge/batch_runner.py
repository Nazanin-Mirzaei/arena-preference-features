"""Command-line batch runner for independent row-level LLM judging."""

from __future__ import annotations

import argparse
import time
from typing import Any

from .config import DEFAULT_MODEL, REQUIRED_COLUMNS, RESULT_COLUMNS
from .parser import parse_json_output
from .prompt import build_judge_prompt
from .utils import clean_text, read_csv_dataset, write_csv, write_jsonl


def validate_row(row: dict[str, str], row_index: int) -> list[str]:
    """Return missing required columns/values for a row."""
    missing = []
    for column in REQUIRED_COLUMNS:
        if not clean_text(row.get(column)):
            missing.append(column)
    if missing:
        print(f"Skipping row {row_index}; missing values: {missing}")
    return missing


def build_jobs(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build one judge job per CSV row."""
    jobs = []
    for row_index, row in enumerate(rows):
        if validate_row(row, row_index):
            continue

        judge_prompt = build_judge_prompt(
            row_id=row["id"],
            user_prompt=row["user_prompt"],
            response_a=row["response_a"],
            response_b=row["response_b"],
        )

        jobs.append(
            {
                "row_index": row_index,
                "id": row["id"],
                "user_prompt": row["user_prompt"],
                "response_a": row["response_a"],
                "response_b": row["response_b"],
                "judge_prompt": judge_prompt,
            }
        )
    return jobs


def flatten_result(
    job: dict[str, Any],
    judge_output: dict[str, Any] | None,
    raw_output: str | None = None,
    parse_error: str | None = None,
) -> dict[str, Any]:
    """Flatten nested judge JSON into CSV-friendly columns."""
    judge_output = judge_output or {}
    eval_a = judge_output.get("response_a_evaluation") or {}
    eval_b = judge_output.get("response_b_evaluation") or {}

    return {
        "id": job.get("id"),
        "user_prompt": job.get("user_prompt"),
        "response_a": job.get("response_a"),
        "response_b": job.get("response_b"),
        "prompt_type": judge_output.get("prompt_type"),
        "educational_field": judge_output.get("educational_field"),
        "correctness_applicable": judge_output.get("correctness_applicable"),
        "response_a_correctness_level": eval_a.get("correctness_level"),
        "response_a_correctness_errors": eval_a.get("correctness_errors"),
        "response_a_needs_external_reference": eval_a.get("needs_external_reference"),
        "response_a_confidence": eval_a.get("confidence"),
        "response_a_rationale": eval_a.get("rationale"),
        "response_b_correctness_level": eval_b.get("correctness_level"),
        "response_b_correctness_errors": eval_b.get("correctness_errors"),
        "response_b_needs_external_reference": eval_b.get("needs_external_reference"),
        "response_b_confidence": eval_b.get("confidence"),
        "response_b_rationale": eval_b.get("rationale"),
        "raw_judge_output": raw_output,
        "parse_error": parse_error,
    }


def run_batch(args: argparse.Namespace) -> None:
    rows = read_csv_dataset(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    jobs = build_jobs(rows)
    print(f"CSV rows read: {len(rows)}")
    print(f"Judge jobs created: {len(jobs)}")

    if args.prompts_output:
        write_jsonl(args.prompts_output, jobs)
        print(f"Saved judge prompts to: {args.prompts_output}")

    if args.prompts_only:
        return

    from openai import APIStatusError, RateLimitError

    from .evaluator import OpenAIJudge

    judge = OpenAIJudge(model=args.model)
    results = []

    for index, job in enumerate(jobs, start=1):
        print(f"Judging row {index}/{len(jobs)} | id={job.get('id')}")
        try:
            raw_output = judge.evaluate_raw(job["judge_prompt"])
            parsed_output, parse_error = parse_json_output(raw_output)
            result = flatten_result(job, parsed_output, raw_output, parse_error)
        except RateLimitError as error:
            print("Stopping because of quota/rate-limit error.")
            print(error)
            break
        except APIStatusError as error:
            print("Stopping because of API status error.")
            print(error)
            break
        except Exception as error:
            result = flatten_result(job, None, None, str(error))

        results.append(result)
        if args.output_jsonl:
            write_jsonl(args.output_jsonl, results)
        time.sleep(args.sleep)

    if args.output_csv:
        write_csv(args.output_csv, results, RESULT_COLUMNS)

    print(f"Completed rows: {len(results)}")
    if args.output_jsonl:
        print(f"Saved JSONL results to: {args.output_jsonl}")
    if args.output_csv:
        print(f"Saved CSV results to: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls.")
    parser.add_argument("--prompts-only", action="store_true", help="Only build prompts, do not call API.")
    parser.add_argument("--prompts-output", default="judge_prompts.jsonl", help="Path for prompt JSONL output.")
    parser.add_argument("--output-jsonl", default="judge_results.jsonl", help="Path for JSONL result output.")
    parser.add_argument("--output-csv", default="judge_results.csv", help="Path for CSV result output.")
    return parser.parse_args()


def main() -> None:
    run_batch(parse_args())


if __name__ == "__main__":
    main()
