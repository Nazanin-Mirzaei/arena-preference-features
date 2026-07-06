"""Configuration constants for the LLM-as-a-Judge pipeline."""

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0

REQUIRED_COLUMNS = ["id", "user_prompt", "response_a", "response_b"]

RESULT_COLUMNS = [
    "id",
    "user_prompt",
    "response_a",
    "response_b",
    "prompt_type",
    "educational_field",
    "correctness_applicable",
    "response_a_correctness_level",
    "response_a_correctness_errors",
    "response_a_needs_external_reference",
    "response_a_confidence",
    "response_a_rationale",
    "response_b_correctness_level",
    "response_b_correctness_errors",
    "response_b_needs_external_reference",
    "response_b_confidence",
    "response_b_rationale",
    "raw_judge_output",
    "parse_error",
]

