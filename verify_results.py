"""Independent score checks, exact search replay, and frozen-weight audit replay."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from run import assert_frozen, ALL_METHODS, FINAL_SEEDS, SEARCH_SEEDS
from energy_config.refined_search import run as replay
from energy_config.data import raw_windows, fit_preprocessing, transform
from world_model.models import build_model


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def close(a, b, label):
    assert np.allclose(a, b, rtol=1e-10, atol=1e-10), (label, a, b)


def energy(power):
    return np.asarray(power, dtype=np.float64).cumsum(axis=1)[:, (1, 3, 5)] / 720


def scores(power, target, ids, scale):
    power = np.asarray(power, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pe, ee = power - target, energy(power) - energy(target)
    rows = []
    for flight in sorted(set(ids.tolist())):
        mask = ids == flight
        r = {"flight": int(flight), "windows": int(mask.sum()), "energy_score": float(np.mean(np.abs(ee[mask])/scale)),
             "power_mae_w": float(np.abs(pe[mask]).mean()), "power_mse_w2": float(np.square(pe[mask]).mean()),
             "power_bias_w": float(pe[mask].mean())}
        for i, h in enumerate((10, 20, 30)):
            r[f"energy_mae_{h}s_wh"] = float(np.abs(ee[mask, i]).mean())
            r[f"energy_bias_{h}s_wh"] = float(ee[mask, i].mean())
        for i in range(6):
            r[f"power_mae_step{i+1}_w"] = float(np.abs(pe[mask, i]).mean())
        rows.append(r)
    result = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("flight", "windows")}
    result["power_rmse_w"] = float(np.sqrt(result.pop("power_mse_w2")))
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "results/development_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    assert_frozen(out)
    protocol, meta = load(out / "protocol.json"), load(out / "provenance.json")
    assert not protocol["original_official_test_windows_loaded"]
    assert not meta["test_windows_constructed_during_fit"]
    assert not set(meta["selection_flights"]) & set(meta["audit_flights"])
    assert set(meta["selection_flights"]) | set(meta["audit_flights"]) == set(meta["validation_flights"])
    assert not (set(meta["training_flights"]) | set(meta["validation_flights"])) & set(meta["test_flights"])
    with np.load(out / "development_inputs.npz") as stored:
        development = {k: stored[k] for k in stored.files}
    with np.load(out / "audit_inputs.npz") as stored:
        audit = {k: stored[k] for k in stored.files}
    assert set(audit["flights"].tolist()) == set(meta["audit_flights"])
    assert not any(k.startswith("test_") for k in development)
    state = fit_preprocessing(raw_windows(BASE / "data", meta["training_flights"]))
    for k, value in state.items():
        close(value, meta["preprocessing_state"][k], "Training-only preprocessing")
    scale = np.array(meta["energy_scale"])
    close(energy(development["train_target"]).std(axis=0), scale, "Training energy scale")
    val = {k[4:]: v for k, v in development.items() if k.startswith("val_")}
    mask = np.isin(val["flights"], meta["selection_flights"])
    selection = {k: v[mask] for k, v in val.items()}
    assert len(selection["flights"]) == meta["selection_windows"]
    runs = []
    for seed in SEARCH_SEEDS:
        for method in ALL_METHODS[1:]:
            run = load(out / "search_runs" / f"{method}_{seed}.json")
            lookup = {(r["config_id"], r["training_seed"], r["epochs"]): r for r in run["events"]}
            assert replay(method, seed, lambda i, s, e: lookup[i, s, e]) == run
            assert sum(r["charged_epochs"] for r in run["events"]) == 320
            assert len({r["config_id"] for r in run["screen_records"]}) == 16
            runs.append(run)
    for seed in SEARCH_SEEDS:
        matching = [r for r in runs if r["search_seed"] == seed]
        assert len({tuple(v["config_id"] for v in r["screen_records"][:4]) for r in matching}) == 1
        a, b = (next(r for r in matching if r["method"] == m) for m in ("random_search", "verified_random"))
        assert [r["config_id"] for r in a["screen_records"]] == [r["config_id"] for r in b["screen_records"]]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensors = {name: {k: torch.from_numpy(data[k]).to(device) for k in ("history", "future")}
                     for name, data in (("selection", selection), ("audit", audit))}
    final_fits = load(out / "full_training/fits.json")
    search_fits = load(out / "search_training/fits.json")
    by_fit = {r["fit_id"]: r for r in final_fits}
    max_diff = 0.0
    with np.load(out / "audit_predictions.npz") as archived:
        for kind, fits in (("search_training", search_fits), ("full_training", final_fits)):
            for r in fits:
                cp = torch.load(out / kind / r["checkpoint"], map_location="cpu", weights_only=True)
                model = build_model("rssm", 21, 19, 64, r["latent_dim"], 5).to(device)
                model.load_state_dict(cp["model_state_dict"])
                assert sum(p.numel() for p in model.parameters()) == r["parameter_count"]
                model.eval()
                with torch.inference_mode():
                    pred = model(input_tensors["selection"]["history"], input_tensors["selection"]["future"], target=None, sample=False)["prediction"]
                    pred = (pred*meta["target_std"]+meta["target_mean"]).cpu().numpy()
                close(scores(pred, selection["target"], selection["flights"], scale)[0]["energy_score"], r["selection_score"], "Selection fitness")
                history = csv_rows(out / kind / Path(r["checkpoint"]).parent / "history.csv")
                minimum = min(history, key=lambda v: float(v["selection_score"]))
                assert int(minimum["epoch"]) == r["best_epoch"]
                assert len(history) == r["epochs_ran"]
                close(float(minimum["selection_score"]), r["selection_score"], "Strict-minimum checkpoint")
                if kind == "full_training":
                    with torch.inference_mode():
                        pred = model(input_tensors["audit"]["history"], input_tensors["audit"]["future"], target=None, sample=False)["prediction"]
                        pred = (pred*meta["target_std"]+meta["target_mean"]).cpu().numpy()
                    expected = archived[f"fit_{r['fit_id']}"]
                    diff = float(np.max(np.abs(pred-expected)))
                    max_diff = max(max_diff, diff)
                    assert np.allclose(pred, expected, atol=0.001, rtol=1e-5)
    selected = load(out / "selected_checkpoints.json")
    metric_rows = csv_rows(out / "audit_metrics.csv")
    per_flight = csv_rows(out / "audit_per_flight.csv")
    identity = lambda r: (r["method"], str(r["search_seed"] or ""), int(r["training_seed"]))
    saved = {identity(r): r for r in metric_rows}
    saved_flights = {(*identity(r), int(r["flight"])): r for r in per_flight}
    recomputed = []
    assert len(selected) == len(metric_rows) == 153
    assert len(per_flight) == 153*20
    with np.load(out / "audit_predictions.npz") as forecasts:
        for item in selected:
            assert item["fit"] == by_fit[item["fit"]["fit_id"]]
            metrics, flights = scores(forecasts[f"fit_{item['fit']['fit_id']}"], audit["target"], audit["flights"], scale)
            k = identity(item)
            for name, value in metrics.items():
                close(value, float(saved[k][name]), "Audit aggregate")
            for row in flights:
                for name, value in row.items():
                    close(value, float(saved_flights[(*k, row["flight"])][name]), "Audit per-flight")
            recomputed.append(dict(item, **metrics))
    summary = load(out / "audit_summary.json")["methods"]
    for method in ALL_METHODS:
        values = [r for r in recomputed if r["method"] == method]
        assert len(values) == (3 if method == "fixed" else 30)
        for k in metrics:
            close(np.mean([r[k] for r in values]), summary[method][k], "Method mean")
        close(np.std([np.mean([r["energy_score"] for r in values if r["training_seed"] == s]) for s in FINAL_SEEDS], ddof=1),
              summary[method]["training_seed_sd"], "Training SD")
        if method != "fixed":
            close(np.std([np.mean([r["energy_score"] for r in values if r["search_seed"] == s]) for s in SEARCH_SEEDS], ddof=1),
                  summary[method]["search_seed_sd"], "Search SD")
        for baseline in ALL_METHODS:
            b = summary[baseline]["energy_score"]
            close(100*(b-summary[method]["energy_score"])/b, summary[method][f"improvement_over_{baseline}_percent"], "Improvement")
    audit_ids = sorted(meta["audit_flights"])
    indices = np.random.default_rng(9511).integers(0, 20, (10000, 20))
    flight_means = {m: np.array([np.mean([float(r["energy_score"]) for r in per_flight if r["method"] == m and int(r["flight"]) == f]) for f in audit_ids]) for m in ALL_METHODS}
    for row in csv_rows(out / "pairwise_comparisons.csv"):
        a, b = (flight_means[row[k]] for k in ("method", "baseline"))
        percent = 100*(b[indices].mean(axis=1)-a[indices].mean(axis=1))/b[indices].mean(axis=1)
        close(np.quantile(percent, (0.025, 0.975)), [float(row["paired_flight_95pct_low"]), float(row["paired_flight_95pct_high"])], "Bootstrap interval")
    accounting = load(out / "fit_accounting.json")
    assert accounting["logical_search_epochs"] == 16000
    assert accounting["actual_unique_search_epochs"] == sum(r["incremental_epochs"] for r in search_fits)
    assert accounting["unique_final_models"] == len(final_fits)
    assert_frozen(out)
    result = {"status": "passed", "search_runs_replayed": len(runs), "validation_checkpoints_checked": len(search_fits)+len(final_fits),
              "audit_comparison_rows_recomputed": len(metric_rows), "audit_per_flight_rows_recomputed": len(per_flight),
              "final_models_reloaded": len(final_fits), "max_reloaded_audit_forecast_difference_w": max_diff,
              "training_only_preprocessing": "passed", "equal_search_budgets": "passed",
              "original_official_test_windows_used": False}
    dump = out / "verification.json"
    dump.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
