import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


DATASET_PATH = Path("data/golden_set.jsonl")
REPORTS_DIR = Path("reports")


def tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def token_recall(expected: str, actual: str) -> float:
    expected_tokens = tokenize(expected)
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & tokenize(actual)) / len(expected_tokens)


def load_dataset(path: Path = DATASET_PATH) -> List[Dict]:
    """
    Read the lab dataset in either JSON array format or JSONL format.
    The current file in this repo is a JSON array, while the original lab
    template describes JSONL, so supporting both keeps the benchmark stable.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return [case for case in parsed if isinstance(case, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    dataset = []
    for line in raw_text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in {"[", "]"}:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(case, dict):
            dataset.append(case)
    return dataset


class ExpertEvaluator:
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()

    async def score(self, case: Dict, response: Dict) -> Dict:
        retrieval = self.retrieval_evaluator.score_response(case, response)
        relevancy = token_recall(case.get("expected_answer", ""), response.get("answer", ""))
        faithfulness = 1.0 if response.get("contexts") and response.get("answer") else 0.0

        return {
            "faithfulness": round(faithfulness, 3),
            "relevancy": round(relevancy, 3),
            "retrieval": {
                "hit_rate": round(retrieval["hit_rate"], 3),
                "mrr": round(retrieval["mrr"], 3),
                "expected_ids": retrieval["expected_ids"],
                "retrieved_ids": retrieval["retrieved_ids"],
            },
        }


async def run_benchmark_with_results(agent_version: str, agent_profile: str):
    print(f"Starting benchmark for {agent_version}...")

    try:
        dataset = load_dataset()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return None, None

    if not dataset:
        print("ERROR: data/golden_set.jsonl is empty or invalid.")
        return None, None

    runner = BenchmarkRunner(MainAgent(profile=agent_profile), ExpertEvaluator(), LLMJudge())
    started_at = time.perf_counter()
    results = await runner.run_all(dataset)
    elapsed = time.perf_counter() - started_at

    total = len(results)
    pass_count = sum(1 for result in results if result["status"] == "pass")
    total_tokens = sum(result.get("metadata", {}).get("tokens_used", 0) for result in results)

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 3),
        },
        "metrics": {
            "avg_score": round(sum(r["judge"]["final_score"] for r in results) / total, 3),
            "pass_rate": round(pass_count / total, 3),
            "hit_rate": round(sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total, 3),
            "mrr": round(sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total, 3),
            "faithfulness": round(sum(r["ragas"]["faithfulness"] for r in results) / total, 3),
            "relevancy": round(sum(r["ragas"]["relevancy"] for r in results) / total, 3),
            "agreement_rate": round(sum(r["judge"]["agreement_rate"] for r in results) / total, 3),
            "avg_latency": round(sum(r["latency"] for r in results) / total, 3),
            "total_tokens": total_tokens,
        },
    }
    return results, summary


async def run_benchmark(version: str, agent_profile: str):
    _, summary = await run_benchmark_with_results(version, agent_profile)
    return summary


async def main():
    v1_summary = await run_benchmark("Agent_V1_Base", "base")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", "optimized")

    if not v1_summary or not v2_summary:
        print("ERROR: Cannot run benchmark. Check data/golden_set.jsonl.")
        return

    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    release_decision = "APPROVE" if delta >= 0 and v2_summary["metrics"]["hit_rate"] >= 0.7 else "BLOCK_RELEASE"

    print("\n--- REGRESSION RESULT ---")
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"Decision: {release_decision}")

    v2_summary["release_gate"] = {
        "decision": release_decision,
        "delta_avg_score": round(delta, 3),
        "minimum_hit_rate": 0.7,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    with (REPORTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with (REPORTS_DIR / "benchmark_results.json").open("w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print("Saved reports/summary.json")
    print("Saved reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
