"""Matched-budget verification and one ordinal-mutation ablation."""
from __future__ import annotations
import numpy as np
from .core import SPACE, initial_ids, ranking, search as original_search

METHODS = ("random_search", "genetic_search", "verified_random", "verified_genetic", "verified_neighbor")


def neighbor_search(seed, lookup):
    records, visited = [], set()

    def query(index, origin):
        index = int(index)
        assert index not in visited
        fit = lookup(index)
        row = {"query": len(records) + 1, "config_id": index,
               "selection_score": fit["selection_score"], "parameter_count": fit["parameter_count"],
               "fit_id": fit["fit_id"], "origin": origin}
        visited.add(index)
        records.append(row)
        return row

    population = [query(i, "shared_initial") for i in initial_ids(seed)]
    rng = np.random.default_rng(seed + 20000)
    fallbacks = attempts_total = local_mutations = global_mutations = 0
    for generation in range(1, 4):
        children = []
        for _ in range(4):
            attempts = 0
            while True:
                attempts += 1
                attempts_total += 1
                if attempts > 200:
                    child = int(rng.choice([i for i in range(len(SPACE)) if i not in visited]))
                    origin = f"generation_{generation}_fallback"
                    fallbacks += 1
                else:
                    def parent():
                        choices = rng.choice(4, 2, replace=False)
                        return min([population[int(i)] for i in choices], key=ranking)["config_id"]
                    a, b = np.array(divmod(parent(), 5)), np.array(divmod(parent(), 5))
                    genes = np.where(rng.random(2) < 0.5, a, b) if rng.random() < 0.8 else a.copy()
                    for gene, size in ((0, 6), (1, 5)):
                        if rng.random() < 0.3:
                            current = int(genes[gene])
                            if rng.random() < 0.8:
                                adjacent = [v for v in (current - 1, current + 1) if 0 <= v < size]
                                genes[gene] = rng.choice(adjacent)
                                local_mutations += 1
                            else:
                                genes[gene] = rng.choice([i for i in range(size) if i != current])
                                global_mutations += 1
                    child = int(genes[0] * 5 + genes[1])
                    origin = f"generation_{generation}"
                if child not in visited:
                    break
            children.append(query(child, origin))
        population = sorted(population + children, key=ranking)[:4]
    assert len(records) == len(visited) == 16
    return {"records": records, "best": min(records, key=ranking),
            "proposal_attempts": attempts_total, "uniform_fallbacks": fallbacks,
            "local_mutation_proposals": local_mutations, "global_mutation_proposals": global_mutations}


def run(method, seed, lookup):
    if method not in METHODS:
        raise ValueError(method)
    events, resource = [], {}

    def query(index, training_seed, epochs, phase):
        previous = resource.get((index, training_seed), 0)
        assert epochs > previous
        fit = lookup(index, training_seed, epochs)
        event = {"config_id": int(index), "training_seed": training_seed, "epochs": epochs,
                 "charged_epochs": epochs - previous, "phase": phase,
                 "selection_score": fit["selection_score"], "parameter_count": fit["parameter_count"],
                 "fit_id": fit["fit_id"]}
        events.append(event)
        resource[index, training_seed] = epochs
        return event

    verified = method.startswith("verified_")
    fidelity = 10 if verified else 20
    callback = lambda i: query(i, 2026, fidelity, "screen")
    if method == "verified_neighbor":
        trace = neighbor_search(seed, callback)
    else:
        proposal = "random_search" if method in ("random_search", "verified_random") else "genetic_search"
        trace = original_search(proposal, seed, callback)
    promoted, finalists = [], []
    if verified:
        for row in sorted(trace["records"], key=ranking)[:4]:
            a = query(row["config_id"], 2026, 20, "extend_A")
            b = query(row["config_id"], 2027, 20, "verify_B")
            promoted.append({"config_id": row["config_id"], "parameter_count": row["parameter_count"],
                             "seed_scores": [a["selection_score"], b["selection_score"]],
                             "selection_score": (a["selection_score"] + b["selection_score"]) / 2})
        for row in sorted(promoted, key=ranking)[:2]:
            c = query(row["config_id"], 2028, 20, "verify_C")
            scores = row["seed_scores"] + [c["selection_score"]]
            finalists.append(dict(row, seed_scores=scores, selection_score=float(np.mean(scores))))
        best = min(finalists, key=ranking)
    else:
        best = trace["best"]
    assert len({r["config_id"] for r in trace["records"]}) == 16
    assert sum(r["charged_epochs"] for r in events) == 320
    return {"method": method, "search_seed": seed, "best_config_id": best["config_id"],
            "best_selection_score": best["selection_score"], "screen_records": trace["records"],
            "events": events, "promoted": promoted, "finalists": finalists, "logical_epochs": 320,
            "proposal_attempts": trace["proposal_attempts"], "uniform_fallbacks": trace["uniform_fallbacks"],
            "local_mutation_proposals": trace.get("local_mutation_proposals", 0),
            "global_mutation_proposals": trace.get("global_mutation_proposals", 0)}
