"""Budget, seed pairing and determinism checks before any neural training."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from energy_config.refined_search import METHODS, run
from energy_config.core import SPACE, initial_ids


def lookup(index, seed, epochs):
    # Synthetic fitness is used only to exercise the search code, never as data.
    dimension, beta = SPACE[index]
    score = 0.3 + abs(dimension - 12) / 100 + abs(index % 5 - 2) / 1000 + (seed % 7) / 10000 + 1 / epochs
    return {"config_id": index, "fit_id": index * 100000 + seed * 10 + epochs,
            "selection_score": score, "parameter_count": dimension * 100}


def main():
    count = 0
    for seed in (11, 22, 33, 44, 55, 66, 77, 88, 99, 110):
        for method in METHODS:
            a, b = run(method, seed, lookup), run(method, seed, lookup)
            assert a == b
            assert [r["config_id"] for r in a["screen_records"][:4]] == initial_ids(seed)
            assert len(set(r["config_id"] for r in a["screen_records"])) == 16
            assert sum(r["charged_epochs"] for r in a["events"]) == 320
            if method.startswith("verified_"):
                assert len(a["promoted"]) == 4 and len(a["finalists"]) == 2
                assert all(len(r["seed_scores"]) == 3 for r in a["finalists"])
            count += 1
    print(json.dumps({"status": "passed", "deterministic_search_runs_checked": count,
                      "purpose": "Code checks with synthetic fitness only; not experimental performance"}, indent=2))


if __name__ == "__main__":
    main()
