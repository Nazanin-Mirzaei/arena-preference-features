
# LLM-as-a-Judge for Independent Correctness Evaluation

This module builds and runs an LLM-as-a-judge pipeline for LMArena-style CSV data.

The evaluation is **row-level** and **independent**:

- each CSV row stays as one row,
- `response_a` and `response_b` are evaluated independently,
- no winner is selected,
- no ranking is produced,
- no comparison between A and B is requested,
- model names are not included in the judge prompt.

The judge receives only:

- `id`
- `user_prompt`
- `response_a`
- `response_b`

and returns structured JSON with prompt classification and independent correctness evaluation for both responses.

## Project Structure

```text
LLM_as_a_Judge/
│
├── README.md
├── requirements.txt
├── __init__.py
│
├── config.py
├── prompt.py
├── parser.py
├── evaluator.py
├── batch_runner.py
└── utils.py
```

## Input Format

The input file must be a CSV file with at least these columns:

```text
id,user_prompt,response_a,response_b
```

Other columns may exist in the CSV, but they are not sent to the LLM judge.

## Output Schema

For each original row, the judge returns one JSON object:

```json
{
  "id": "...",
  "prompt_type": "educational",
  "educational_field": "math",
  "correctness_applicable": true,
  "response_a_evaluation": {
    "correctness_level": "fully_correct",
    "correctness_errors": [],
    "needs_external_reference": false,
    "confidence": 0.92,
    "rationale": "Short explanation for response A only."
  },
  "response_b_evaluation": {
    "correctness_level": "incorrect",
    "correctness_errors": ["logical_error"],
    "needs_external_reference": false,
    "confidence": 0.88,
    "rationale": "Short explanation for response B only."
  }
}
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
set OPENAI_API_KEY=your_api_key
```

On macOS/Linux:

```bash
export OPENAI_API_KEY=your_api_key
```

## Usage

Run the full pipeline:

```bash
python -m LLM_as_a_Judge.batch_runner \
  --input path/to/input.csv \
  --output-jsonl judge_results.jsonl \
  --output-csv judge_results.csv
```

Use a different model:

```bash
python -m LLM_as_a_Judge.batch_runner \
  --input path/to/input.csv \
  --model gpt-4.1-mini \
  --output-jsonl judge_results.jsonl \
  --output-csv judge_results.csv
```

Build prompts only, without calling the API:

```bash
python -m LLM_as_a_Judge.batch_runner \
  --input path/to/input.csv \
  --prompts-only \
  --prompts-output judge_prompts.jsonl
```

Limit the run for testing:

```bash
python -m LLM_as_a_Judge.batch_runner \
  --input path/to/input.csv \
  --limit 5 \
  --output-jsonl judge_results_test.jsonl \
  --output-csv judge_results_test.csv
```

## Python Usage

```python
from LLM_as_a_Judge.prompt import build_judge_prompt
from LLM_as_a_Judge.evaluator import OpenAIJudge

prompt = build_judge_prompt(
    row_id="example-1",
    user_prompt="What is 2 + 2?",
    response_a="The answer is 4.",
    response_b="The answer is 5."
)

judge = OpenAIJudge(model="gpt-4.1-mini")
result = judge.evaluate(prompt)
print(result)
```

## Methodological Notes

This module is designed for independent correctness annotation. Even though both responses are present in the same judge prompt, the instructions explicitly require the LLM judge to evaluate them as two separate exam answers.

The judge is instructed:

- not to compare A and B,
- not to choose a winner,
- not to rank responses,
- not to let one response influence the other,
- to behave in an order-invariant way.

This design is different from pairwise preference judging.
