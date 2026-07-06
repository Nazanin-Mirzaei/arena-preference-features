"""Judge prompt template and prompt builder."""

from __future__ import annotations

from .utils import clean_text


JUDGE_PROMPT_TEMPLATE = """
You are a strict LLM-as-a-judge evaluator.

You will receive one USER PROMPT and two assistant responses: RESPONSE_A and RESPONSE_B.

Your task is NOT to compare the two responses.
Your task is NOT to choose a winner.
Your task is NOT to rank the responses.
Your task is NOT to decide which response is better.

Instead, evaluate RESPONSE_A and RESPONSE_B as two completely independent exam answers to the same question.

Core rule:
- Evaluate RESPONSE_A as if RESPONSE_B did not exist.
- Evaluate RESPONSE_B as if RESPONSE_A did not exist.
- Do not let any information, style, quality, correctness, or error in one response influence the evaluation of the other response.
- The output must be order-invariant: swapping RESPONSE_A and RESPONSE_B should not change the judgment of either response except for the field names.
- Do not use cross-response comparison.
- Do not mention winner, loser, tie, ranking, preference, or relative quality.

You must perform these tasks:
1. Classify the USER PROMPT type.
2. If the USER PROMPT is educational, classify its educational field.
3. Decide whether correctness is applicable to this prompt.
4. Independently evaluate correctness for RESPONSE_A.
5. Independently evaluate correctness for RESPONSE_B.

Prompt type labels:
- educational
- chitchatting
- translation_language
- creative_writing
- code
- summarization
- fact_checking
- reasoning
- advice
- data_analysis
- other

Educational field labels:
- math
- physics
- computer_science
- biology
- chemistry
- history
- engineering
- literature
- language_learning
- medicine_health
- economics_business
- philosophy
- law
- psychology
- other_education

If prompt_type is not educational, educational_field must be null.

Correctness applicability:
Set correctness_applicable to true if the prompt has an objective or verifiable answer, including math, factual questions, code, logical reasoning, translation accuracy, extraction from provided text, constrained summarization, or explicit formatting constraints.

Set correctness_applicable to false for subjective, creative, emotional, preference-based, open-ended, or style-only prompts.

Correctness level labels:
- fully_correct
- partially_correct
- incorrect
- unknown
- null

Use null only when correctness_applicable is false.

Allowed correctness_errors:
- factual_error
- logical_error
- computational_error
- code_error
- translation_error
- misinterpretation
- missing_info
- format_error
- unsupported_claim
- other

Return valid JSON only, exactly in this structure:
{
  "id": "{id}",
  "prompt_type": "educational|chitchatting|translation_language|creative_writing|code|summarization|fact_checking|reasoning|advice|data_analysis|other",
  "educational_field": "math|physics|computer_science|biology|chemistry|history|engineering|literature|language_learning|medicine_health|economics_business|philosophy|law|psychology|other_education|null",
  "correctness_applicable": true,
  "response_a_evaluation": {
    "correctness_level": "fully_correct|partially_correct|incorrect|unknown|null",
    "correctness_errors": [],
    "needs_external_reference": false,
    "confidence": 0.0,
    "rationale": "One or two short sentences evaluating only RESPONSE_A."
  },
  "response_b_evaluation": {
    "correctness_level": "fully_correct|partially_correct|incorrect|unknown|null",
    "correctness_errors": [],
    "needs_external_reference": false,
    "confidence": 0.0,
    "rationale": "One or two short sentences evaluating only RESPONSE_B."
  }
}

INPUT

ID:
{id}

USER PROMPT:
<<<USER_PROMPT>>>
{user_prompt}
<<<END_USER_PROMPT>>>

RESPONSE_A:
<<<RESPONSE_A>>>
{response_a}
<<<END_RESPONSE_A>>>

RESPONSE_B:
<<<RESPONSE_B>>>
{response_b}
<<<END_RESPONSE_B>>>
""".strip()


def build_judge_prompt(
    row_id: str,
    user_prompt: str,
    response_a: str,
    response_b: str,
) -> str:
    """Build one independent row-level judge prompt."""
    values = {
        "id": clean_text(row_id),
        "user_prompt": clean_text(user_prompt),
        "response_a": clean_text(response_a),
        "response_b": clean_text(response_b),
    }

    prompt = JUDGE_PROMPT_TEMPLATE
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt

