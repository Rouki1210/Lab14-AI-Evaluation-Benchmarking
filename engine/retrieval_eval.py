from typing import Dict, List


class RetrievalEvaluator:
    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3,
    ) -> float:
        """
        Hit Rate@K = 1 if at least one expected document id appears in top K.
        """
        expected = {str(doc_id) for doc_id in expected_ids}
        retrieved_top_k = [str(doc_id) for doc_id in retrieved_ids[:top_k]]
        return 1.0 if expected and any(doc_id in expected for doc_id in retrieved_top_k) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        MRR = 1 / rank of the first relevant retrieved document.
        Rank is 1-indexed. If no expected id is retrieved, MRR is 0.
        """
        expected = {str(doc_id) for doc_id in expected_ids}
        if not expected:
            return 0.0

        for index, doc_id in enumerate(retrieved_ids, start=1):
            if str(doc_id) in expected:
                return 1.0 / index
        return 0.0

    def score_response(self, case: Dict, response: Dict, top_k: int = 3) -> Dict:
        expected_ids = [str(item) for item in case.get("expected_retrieval_ids", [])]
        retrieved_ids = [str(item) for item in response.get("retrieved_ids", [])]

        if not expected_ids:
            has_context = bool(response.get("contexts"))
            return {
                "hit_rate": 1.0 if has_context else 0.0,
                "mrr": 1.0 if has_context else 0.0,
                "expected_ids": [],
                "retrieved_ids": retrieved_ids,
            }

        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
        }

    async def evaluate_batch(self, dataset: List[Dict], agent, top_k: int = 3) -> Dict:
        scores = []

        for case in dataset:
            response = await agent.query(case["question"])
            retrieval_score = self.score_response(case, response, top_k=top_k)
            scores.append(
                {
                    "question": case["question"],
                    **retrieval_score,
                }
            )

        if not scores:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "details": []}

        return {
            "avg_hit_rate": sum(item["hit_rate"] for item in scores) / len(scores),
            "avg_mrr": sum(item["mrr"] for item in scores) / len(scores),
            "details": scores,
        }