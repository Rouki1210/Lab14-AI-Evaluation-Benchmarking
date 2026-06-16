import asyncio
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI

class LLMJudge:
    """
    Multi-judge evaluator using two OpenAI-compatible model configs from .env:
    - OPEN_AI_KEY, OPEN_BASE_URL, OPEN_AI_MODEL
    - AI_KEY, AI_BASE_URL, AI_MODEL
    """

    def __init__(self):
        load_dotenv(".env", override=True)
        self.judges = [
            self._build_judge_config(
                name="judge_open_ai",
                api_key_env="OPEN_AI_KEY",
                base_url_env="OPEN_BASE_URL",
                model_env="OPEN_AI_MODEL",
            ),
            self._build_judge_config(
                name="judge_ai",
                api_key_env="AI_KEY",
                base_url_env="AI_BASE_URL",
                model_env="AI_MODEL",
            ),
        ]
        self.rubrics = {
            "accuracy": "1-5 based on factual correctness against the expected answer.",
            "grounding": "1-5 based on whether the answer is supported by the provided ground truth.",
            "completeness": "1-5 based on whether the key legal details are covered.",
        }

    def _build_judge_config(
        self,
        name: str,
        api_key_env: str,
        base_url_env: str,
        model_env: str,
    ) -> Dict[str, Any]:
        api_key = os.getenv(api_key_env)
        base_url = os.getenv(base_url_env)
        model = os.getenv(model_env)

        client = None
        if api_key and model:
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = AsyncOpenAI(**client_kwargs)

        return {
            "name": name,
            "client": client,
            "model": model,
            "api_key_env": api_key_env,
            "base_url_env": base_url_env,
            "model_env": model_env,
        }

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        raw_results = await asyncio.gather(
            *[
                self._evaluate_with_judge(judge, question, answer, ground_truth)
                for judge in self.judges
            ],
            return_exceptions=True,
        )
        failures = [
            f"{judge['name']} ({judge['model']}): {result}"
            for judge, result in zip(self.judges, raw_results)
            if isinstance(result, Exception)
        ]
        if failures:
            raise RuntimeError(
                "Real LLM judge failed; no mock/fallback is enabled. "
                + " | ".join(failures)
            )

        judge_results = [result for result in raw_results if isinstance(result, dict)]

        individual_scores = {
            result["judge"]: result["score"]
            for result in judge_results
        }
        score_values = list(individual_scores.values())
        max_delta = max(score_values) - min(score_values) if score_values else 0.0
        agreement_rate = 1.0 if max_delta <= 0.5 else 0.5 if max_delta <= 1.0 else 0.0

        if max_delta > 1.0:
            # Conservative conflict handling: when judges disagree sharply, use
            # the lower score instead of over-trusting the average.
            final_score = min(score_values)
            conflict_resolution = "large_disagreement_use_lower_score"
        else:
            final_score = sum(score_values) / len(score_values)
            conflict_resolution = "average_scores"

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": agreement_rate,
            "individual_scores": individual_scores,
            "judge_details": judge_results,
            "conflict_resolution": conflict_resolution,
            "reasoning": "Two configured judges evaluated accuracy, grounding, and completeness.",
        }

    async def _evaluate_with_judge(
        self,
        judge: Dict[str, Any],
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        if judge["client"] is None:
            raise RuntimeError(
                f"Missing judge config for {judge['name']}: "
                f"{judge['api_key_env']} and {judge['model_env']} are required."
            )

        prompt = self._build_prompt(question, answer, ground_truth)

        completion = await judge["client"].chat.completions.create(
            model=judge["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator for Vietnamese legal QA. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        parsed = json.loads(content)
        score = float(parsed.get("score", 0))
        score = max(1.0, min(5.0, score))
        return {
            "judge": judge["name"],
            "model": judge["model"],
            "score": score,
            "reasoning": str(parsed.get("reasoning", "")),
            "used_fallback": False,
        }

    def _build_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        return f"""
Evaluate the agent answer against the expected answer.

Rubric:
- 5: Fully correct, grounded, complete.
- 4: Mostly correct with minor missing detail.
- 3: Partially correct but missing important detail.
- 2: Mostly incorrect or poorly grounded.
- 1: Incorrect, unsupported, or refuses when it should answer.

Return JSON exactly like:
{{"score": 4, "reasoning": "short explanation"}}

Question:
{question}

Expected answer:
{ground_truth}

Agent answer:
{answer}
""".strip()

    async def check_position_bias(self, response_a: str, response_b: str):
        return {
            "implemented": False,
            "note": "Position-bias checks can be added by swapping A/B answer order in pairwise judging.",
        }
