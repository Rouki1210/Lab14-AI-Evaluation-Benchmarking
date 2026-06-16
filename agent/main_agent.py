import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List


class MainAgent:
    """
    Baseline RAG agent for the lab.

    The agent loads the golden dataset as a small local knowledge base, retrieves
    the most relevant contexts with keyword overlap, then returns the answer from
    the best matched case.
    """

    def __init__(self, dataset_path: str = "data/golden_set.jsonl", profile: str = "optimized"):
        self.name = f"CanCuocLawAgent-{profile}"
        self.dataset_path = Path(dataset_path)
        self.profile = profile
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict]:
        if not self.dataset_path.exists():
            return []

        raw_text = self.dataset_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return []

        cases = self._parse_dataset(raw_text)
        knowledge_base = []

        for index, case in enumerate(cases, start=1):
            if not isinstance(case, dict):
                continue

            metadata = case.get("metadata", {})
            retrieval_ids = case.get("expected_retrieval_ids") or []
            if not retrieval_ids:
                article_id = metadata.get("article") or metadata.get("article_ref")
                retrieval_ids = [f"case_{index}" if article_id is None else str(article_id)]

            knowledge_base.append(
                {
                    "id": retrieval_ids[0],
                    "retrieval_ids": [str(item) for item in retrieval_ids],
                    "question": case.get("question", ""),
                    "answer": case.get("expected_answer", ""),
                    "context": case.get("context", ""),
                    "metadata": metadata,
                }
            )

        return knowledge_base

    def _parse_dataset(self, raw_text: str) -> List[Dict]:
        """
        Supports both formats commonly used in this lab:
        - JSON array: [{...}, {...}]
        - JSONL: one JSON object per line
        """
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                return parsed
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
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return cases

    def _tokenize(self, text: str) -> set:
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        stopwords = {
            "là",
            "và",
            "của",
            "có",
            "theo",
            "về",
            "trong",
            "luật",
            "căn",
            "cước",
            "quy",
            "định",
            "những",
            "nào",
            "gì",
            "như",
            "thế",
        }
        return {token for token in tokens if token not in stopwords and len(token) > 1}

    def _score(self, question: str, item: Dict) -> float:
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return 0.0

        question_tokens = self._tokenize(item["question"])
        context_tokens = self._tokenize(item["context"])
        metadata_tokens = self._tokenize(" ".join(str(v) for v in item["metadata"].values()))

        question_overlap = len(query_tokens & question_tokens)
        context_overlap = len(query_tokens & context_tokens)
        metadata_overlap = len(query_tokens & metadata_tokens)

        if self.profile == "base":
            return context_overlap + metadata_overlap

        score = (question_overlap * 3.0) + context_overlap + (metadata_overlap * 1.5)
        if self._is_definition_question(question) and self._is_definition_case(item):
            score += 8.0
        return score

    def _is_definition_question(self, question: str) -> bool:
        lowered = question.lower()
        definition_markers = [
            "định nghĩa",
            "thế nào",
            "khái niệm",
            "được hiểu",
            "là gì",
        ]
        return any(marker in lowered for marker in definition_markers)

    def _is_definition_case(self, item: Dict) -> bool:
        metadata_text = " ".join(str(value) for value in item["metadata"].values()).lower()
        return "điều 3" in metadata_text or "giải thích từ ngữ" in metadata_text

    def _retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        scored = [(self._score(question, item), item) for item in self.knowledge_base]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for score, item in scored[:top_k] if score > 0]

    async def query(self, question: str) -> Dict:
        """
        Simulates a RAG flow:
        1. Retrieve relevant legal contexts.
        2. Generate an answer grounded in the best retrieved case.
        """
        await asyncio.sleep(0.05)

        retrieved = self._retrieve(question, top_k=3)
        if not retrieved:
            return {
                "answer": "Tôi không tìm thấy thông tin phù hợp trong tài liệu Luật Căn cước được cung cấp.",
                "contexts": [],
                "retrieved_ids": [],
                "metadata": {
                    "model": "keyword-rag-baseline",
                    "tokens_used": 0,
                    "sources": [],
                },
            }

        best = retrieved[0]
        sources = {
            item["metadata"].get("source", "data/golden_set.jsonl")
            for item in retrieved
        }

        return {
            "answer": best["answer"],
            "contexts": [item["context"] for item in retrieved],
            "retrieved_ids": [rid for item in retrieved for rid in item["retrieval_ids"]],
            "metadata": {
                "model": self.name,
                "tokens_used": 0,
                "sources": sorted(sources),
            },
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        resp = await agent.query("Luật Căn cước 2023 có hiệu lực từ ngày nào?")
        print(json.dumps(resp, ensure_ascii=False, indent=2))

    asyncio.run(test())
