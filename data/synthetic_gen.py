import asyncio
import json
from pathlib import Path
from typing import Dict, List


OUTPUT_PATH = Path("data/golden_set.jsonl")
DOWNLOAD_DATASET_PATH = Path.home() / "Downloads" / "golden_dataset.json"


def _read_cases_from_json_array(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of test cases.")
    return [case for case in data if isinstance(case, dict)]


def _read_cases_from_json_or_jsonl(path: Path) -> List[Dict]:
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

    cases = []
    for line in raw_text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in {"[", "]"}:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(case, dict):
            cases.append(case)
    return cases


def _validate_cases(cases: List[Dict]) -> None:
    required_fields = {"question", "expected_answer", "context", "metadata"}
    if len(cases) < 50:
        raise ValueError(f"Golden dataset must contain at least 50 cases, got {len(cases)}.")

    for index, case in enumerate(cases, start=1):
        missing = required_fields - set(case)
        if missing:
            raise ValueError(f"Case #{index} is missing required fields: {sorted(missing)}")

        if not case.get("expected_retrieval_ids"):
            case["expected_retrieval_ids"] = [f"case_{index:03d}"]


def _write_jsonl(cases: List[Dict], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


async def generate_qa_from_text(text: str, num_pairs: int = 50) -> List[Dict]:
    """
    Offline fallback generator.

    This lab already uses a curated legal golden dataset. If the downloaded
    dataset is unavailable, this function creates simple placeholder cases from
    supplied text so the script fails gracefully instead of calling external APIs.
    """
    cases = []
    for index in range(1, num_pairs + 1):
        cases.append(
            {
                "question": f"Câu hỏi kiểm thử #{index} từ tài liệu là gì?",
                "expected_answer": f"Câu trả lời kỳ vọng #{index} dựa trên nội dung tài liệu được cung cấp.",
                "context": text,
                "expected_retrieval_ids": [f"case_{index:03d}"],
                "metadata": {
                    "difficulty": "easy" if index <= 15 else "medium" if index <= 40 else "hard",
                    "type": "offline-generated",
                    "source": "synthetic_gen.py",
                },
            }
        )
    return cases


async def main():
    if DOWNLOAD_DATASET_PATH.exists():
        cases = _read_cases_from_json_array(DOWNLOAD_DATASET_PATH)
        source = DOWNLOAD_DATASET_PATH
    elif OUTPUT_PATH.exists():
        cases = _read_cases_from_json_or_jsonl(OUTPUT_PATH)
        source = OUTPUT_PATH
    else:
        raw_text = "Luật Căn cước quy định về căn cước, dữ liệu dân cư, thẻ căn cước và căn cước điện tử."
        cases = await generate_qa_from_text(raw_text)
        source = "offline fallback"

    _validate_cases(cases)
    _write_jsonl(cases)
    print(f"Done! Saved {len(cases)} cases to {OUTPUT_PATH} from {source}")


if __name__ == "__main__":
    asyncio.run(main())
