"""
RAGAS-based evaluation of the RAG pipeline.
Metrics: answer_relevancy, faithfulness, context_precision, context_recall.

Usage:
    python src/evaluate.py --index_path data/faiss_index --qa_path data/eval_qa.json
"""

import argparse
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from chain import RAGChain


def load_qa_pairs(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def run_ragas_eval(qa_pairs: list, chain: RAGChain) -> dict:
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in qa_pairs:
        result = chain.ask(item["question"])
        data["question"].append(item["question"])
        data["answer"].append(result["answer"])
        data["contexts"].append([result["context_used"]])
        data["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(data)
    scores = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
    )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default="data/faiss_index")
    parser.add_argument("--qa_path", type=str, default="data/eval_qa.json")
    args = parser.parse_args()

    chain = RAGChain(index_path=args.index_path)
    qa_pairs = load_qa_pairs(args.qa_path)

    print(f"Evaluating {len(qa_pairs)} QA pairs with RAGAS...")
    scores = run_ragas_eval(qa_pairs, chain)

    print("\nRAGAS Evaluation Results:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()
