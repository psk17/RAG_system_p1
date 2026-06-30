import json
from pathlib import Path
from rag_system.evaluation.ragas_runner import evaluate_rag

def benchmark():
    current_dir = Path(__file__).parent
    dataset_path = current_dir / "datasets" / "gold_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = evaluate_rag(dataset)
    print(results)

if __name__ == "__main__":
    benchmark()
