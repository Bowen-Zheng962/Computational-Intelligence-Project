"""Fresh development experiment: fit on selection flights, then frozen audit once."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
import numpy as np
import torch
from world_model.models import build_model
from energy_config.core import SPACE
from energy_config.data import development
from energy_config.metrics import evaluate, sha256
from energy_config.progressive import ProgressiveTrainer
from energy_config.training import Trainer
from energy_config.refined_search import METHODS, run as search

SEARCH_SEEDS = tuple(11 * i for i in range(1, 11))
FINAL_SEEDS = (511, 522, 533)
ALL_METHODS = ("fixed",) + METHODS


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def code_hashes():
    return {p.relative_to(BASE).as_posix(): sha256(p) for p in [BASE / "run.py"] + sorted((BASE / "src").rglob("*.py"))}


def fit(output, device):
    if any((output / n).exists() for n in ("frozen_manifest.json", "audit_summary.json")):
        raise RuntimeError("Frozen output protected; use another output directory.")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "purpose": "Post-original-test development experiment, not a new independent official test",
        "methods": list(ALL_METHODS), "space": [list(v) for v in SPACE],
        "search_seeds": list(SEARCH_SEEDS), "search_training_seeds": [2026, 2027, 2028],
        "final_training_seeds": list(FINAL_SEEDS), "split_seed": 2041, "selection_flights": 19, "audit_flights": 20,
        "base_training_flights": 118, "normalization": "original training flights only",
        "logical_epochs_per_search": 320, "distinct_candidates_per_search": 16,
        "ordinary_search": "16 candidates x20 epochs, training seed 2026",
        "verified_search": "16x10 + 4x(10 extension + 20 seed-B) + 2x20 seed-C = 320",
        "neighbor_mutation": "mutation rate 0.3 per gene; conditional 0.8 adjacent-category / 0.2 global-category",
        "genetic_settings": "4 shared initial candidates; population 4; 3x4 offspring; binary tournaments; crossover 0.8; elitist top4",
        "fitness": "selection-flight macro mean normalized energy MAE at 10/20/30 seconds",
        "full_epochs": 80, "patience": 10, "progress_min_delta": 0.0001,
        "audit_policy": "all methods, seeds and checkpoints frozen before first audit scoring; no audit-guided retuning",
        "original_official_test_windows_loaded": False, "device": device,
        "code_sha256": code_hashes(),
        "data_sha256": {n: sha256(BASE / "data" / n) for n in ("wm_segments.csv", "wm_split.csv", "preprocess_config.json")},
    }
    if (output / "protocol.json").exists():
        assert load(output / "protocol.json") == protocol
    else:
        dump(output / "protocol.json", protocol)
    began = time.perf_counter()
    datasets, meta = development(BASE / "data", output)
    validation = np.array(sorted(meta["validation_flights"]))
    shuffled = np.random.default_rng(2041).permutation(validation)
    selection_ids = sorted(shuffled[:19].tolist())
    audit_ids = sorted(shuffled[19:].tolist())
    mask = np.isin(datasets["val"]["flights"], selection_ids)
    assert not set(selection_ids) & set(audit_ids)
    assert len(set(selection_ids) | set(audit_ids)) == 39
    meta.update(device=device, selection_flights=selection_ids, audit_flights=audit_ids,
                selection_windows=int(mask.sum()), audit_windows=int((~mask).sum()),
                scope=protocol["purpose"], torch_version=str(torch.__version__),
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None)
    dump(output / "provenance.json", meta)
    for name in ("search_training", "full_training", "search_runs"):
        (output / name).mkdir(exist_ok=True)
    screening = ProgressiveTrainer(output / "search_training", datasets, mask, meta, build_model)
    final = Trainer(output / "full_training", datasets, mask, meta, build_model)
    print(f"FIT: train={len(datasets['train']['flights'])}; selection={mask.sum()} windows/{len(selection_ids)} flights; audit not scored", flush=True)
    runs = []
    for seed in SEARCH_SEEDS:
        for method in METHODS:
            path = output / "search_runs" / f"{method}_{seed}.json"
            if path.exists():
                result = load(path)
            else:
                result = search(method, seed, lambda i, s, e: screening.fit(i, s, e, 0, "search")[0])
                dump(path, result)
            runs.append(result)
            print(f"SEARCH {method} seed={seed} winner={result['best_config_id']} score={result['best_selection_score']:.6f} logical_epochs=320", flush=True)
    choices = [{"method": "fixed", "search_seed": None, "config_id": 17}] + [
        {"method": r["method"], "search_seed": r["search_seed"], "config_id": r["best_config_id"]} for r in runs]
    dump(output / "search_choices_frozen.json", choices)
    selected = []
    for choice in choices:
        for seed in FINAL_SEEDS:
            record, reused = final.fit(choice["config_id"], seed, 80, 10, "final_" + choice["method"])
            selected.append(dict(choice, training_seed=seed, fit=record, cache_reused=reused))
    dump(output / "selected_checkpoints.json", selected)
    accounting = {"fit_elapsed_seconds": time.perf_counter()-began,
                  "logical_search_epochs": sum(r["logical_epochs"] for r in runs),
                  "actual_unique_search_epochs": sum(r["incremental_epochs"] for r in screening.all_fits),
                  "unique_search_trajectories": len({(r["config_id"], r["seed"]) for r in screening.all_fits}),
                  "search_checkpoint_rungs": len(screening.all_fits), "logical_final_slots": len(selected),
                  "unique_final_models": len(final.all_fits), "actual_final_epochs": sum(r["epochs_ran"] for r in final.all_fits),
                  "audit_scores_computed": False, "original_official_test_used": False}
    dump(output / "fit_accounting.json", accounting)
    assert code_hashes() == protocol["code_sha256"]
    dump(output / "frozen_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": code_hashes(),
        "files_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "audit_scored_when_frozen": False, "original_official_test_used": False})
    print(f"FROZEN {len(selected)} logical positions/{len(final.all_fits)} unique final models", flush=True)


def assert_frozen(output):
    manifest = load(output / "frozen_manifest.json")
    for name, expected in manifest["code_sha256"].items():
        assert sha256(BASE / name) == expected, name
    for name, expected in manifest["files_sha256"].items():
        assert sha256(output / name) == expected, name
    return manifest


def audit(output, device):
    if (output / "audit_summary.json").exists():
        print("Audit already exists; no repeated audit evaluation.", flush=True)
        return
    assert_frozen(output)
    meta = load(output / "provenance.json")
    selected = load(output / "selected_checkpoints.json")
    fits = load(output / "full_training/fits.json")
    with np.load(output / "development_inputs.npz") as stored:
        validation = {k[4:]: stored[k] for k in stored.files if k.startswith("val_")}
    mask = np.isin(validation["flights"], meta["audit_flights"])
    data = {k: v[mask] for k, v in validation.items()}
    assert len(set(data["flights"].tolist())) == 20
    history, future = (torch.from_numpy(data[k]).to(device) for k in ("history", "future"))
    forecasts = {}
    for row in fits:
        cp = torch.load(output / "full_training" / row["checkpoint"], map_location="cpu", weights_only=True)
        model = build_model("rssm", history.shape[-1], future.shape[-1], 64, row["latent_dim"], 5).to(device)
        model.load_state_dict(cp["model_state_dict"])
        model.eval()
        with torch.inference_mode():
            power = model(history, future, target=None, sample=False)["prediction"]
            power = (power * meta["target_std"] + meta["target_mean"]).cpu().numpy()
        assert np.isfinite(power).all()
        forecasts[f"fit_{row['fit_id']}"] = power
    rows, flight_rows = [], []
    for item in selected:
        metrics, byflight = evaluate(forecasts[f"fit_{item['fit']['fit_id']}"], data["target"], data["flights"], np.array(meta["energy_scale"]))
        identity = {k: item[k] for k in ("method", "search_seed", "training_seed", "config_id")}
        identity["fit_id"] = item["fit"]["fit_id"]
        rows.append(dict(identity, **metrics))
        flight_rows.extend(dict(identity, **r) for r in byflight)
    summary = {"scope": "Development audit only; original official test untouched by this new experiment",
               "audit_flights": 20, "audit_windows": len(data["flights"]), "logical_comparison_positions": len(rows),
               "unique_final_models": len(fits), "methods": {}}
    excluded = {"method", "search_seed", "training_seed", "config_id", "fit_id"}
    for method in ALL_METHODS:
        values = [r for r in rows if r["method"] == method]
        metrics = {k: float(np.mean([r[k] for r in values])) for k in values[0] if k not in excluded}
        metrics["training_seed_sd"] = float(np.std([np.mean([r["energy_score"] for r in values if r["training_seed"] == s]) for s in FINAL_SEEDS], ddof=1))
        metrics["search_seed_sd"] = None if method == "fixed" else float(np.std([np.mean([r["energy_score"] for r in values if r["search_seed"] == s]) for s in SEARCH_SEEDS], ddof=1))
        metrics["mean_parameters"] = float(np.mean([r["fit"]["parameter_count"] for r in selected if r["method"] == method]))
        summary["methods"][method] = metrics
    for metrics in summary["methods"].values():
        for baseline in ALL_METHODS:
            b = summary["methods"][baseline]["energy_score"]
            metrics[f"improvement_over_{baseline}_percent"] = 100 * (b-metrics["energy_score"]) / b
    ids = sorted(meta["audit_flights"])
    samples = np.random.default_rng(9511).integers(0, 20, (10000, 20))
    flight_means = {m: np.array([np.mean([r["energy_score"] for r in flight_rows if r["method"] == m and r["flight"] == f]) for f in ids]) for m in ALL_METHODS}
    pairs = []
    for method in ALL_METHODS:
        for baseline in ALL_METHODS:
            if method == baseline:
                continue
            a, b = flight_means[method], flight_means[baseline]
            relative = 100 * (b[samples].mean(axis=1)-a[samples].mean(axis=1))/b[samples].mean(axis=1)
            lo, hi = np.quantile(relative, [0.025, 0.975])
            pairs.append({"method": method, "baseline": baseline,
                          "relative_improvement_percent": 100*(b.mean()-a.mean())/b.mean(),
                          "paired_flight_95pct_low": float(lo), "paired_flight_95pct_high": float(hi),
                          "flights_with_lower_error": int((a < b).sum()), "audit_flights": 20,
                          "exploratory_no_multiplicity_adjustment": True})
    assert_frozen(output)
    np.savez_compressed(output / "audit_inputs.npz", **data)
    np.savez_compressed(output / "audit_predictions.npz", **forecasts)
    write_csv(output / "audit_metrics.csv", rows)
    write_csv(output / "audit_per_flight.csv", flight_rows)
    write_csv(output / "pairwise_comparisons.csv", pairs)
    dump(output / "audit_summary.json", summary)
    print(json.dumps({m: {k: v[k] for k in ("energy_score", "improvement_over_fixed_percent", "improvement_over_random_search_percent", "improvement_over_verified_random_percent")}
                      for m, v in summary["methods"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "audit", "all"), required=True)
    parser.add_argument("--output", type=Path, default=BASE / "results/development_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(4)
    if args.stage in ("fit", "all"):
        fit(args.output.resolve(), device)
    if args.stage in ("audit", "all"):
        audit(args.output.resolve(), device)


if __name__ == "__main__":
    main()
