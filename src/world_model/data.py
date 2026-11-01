from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


EARTH_RADIUS_M = 6_371_000.0

HISTORY_FEATURES = [
    "integration_duration_s",
    "local_east_m",
    "local_north_m",
    "relative_altitude_m",
    "mean_ground_speed_mps",
    "mean_vertical_speed_mps",
    "delta_altitude_m",
    "route_heading_east",
    "route_heading_north",
    "wind_speed_mean",
    "wind_speed_std",
    "wind_east_mps",
    "wind_north_mps",
    "wind_along_route_mps",
    "wind_cross_route_mps",
    "mean_battery_voltage_V",
    "mean_battery_current_A",
    "mean_power_W",
    "cumulative_energy_Wh",
    "programmed_speed_mps",
    "payload_g",
]

FUTURE_FEATURES = [
    "eta_s",
    "planned_duration_s",
    "local_east_m",
    "local_north_m",
    "relative_altitude_m",
    "route_delta_east_m",
    "route_delta_north_m",
    "route_delta_altitude_m",
    "planned_ground_speed_mps",
    "planned_vertical_speed_mps",
    "route_heading_east",
    "route_heading_north",
    "programmed_speed_mps",
    "payload_g",
    "forecast_wind_speed_mps",
    "forecast_wind_east_mps",
    "forecast_wind_north_mps",
    "forecast_wind_along_route_mps",
    "forecast_wind_cross_route_mps",
]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    history_dim: int
    future_dim: int
    target_mean: float
    target_std: float
    history_feature_names: list[str]
    future_feature_names: list[str]
    description: str
    step_seconds: float
    history_steps: int
    horizon_steps: int
    preprocessing_state: dict[str, Any]


class WindowDataset(Dataset):
    def __init__(self, windows: list[dict[str, np.ndarray | int]]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        return self.windows[index]


def collate_windows(batch: list[dict[str, np.ndarray | int]]) -> dict[str, torch.Tensor]:
    array_keys = [
        "history",
        "future",
        "y",
        "physics",
        "dt",
        "lead_time",
        "target_segment_index",
    ]
    result: dict[str, torch.Tensor] = {}
    for key in array_keys:
        values = np.stack([np.asarray(item[key]) for item in batch])
        if key == "target_segment_index":
            result[key] = torch.from_numpy(values.astype(np.int64))
        else:
            result[key] = torch.from_numpy(values.astype(np.float32))
    result["flight_id"] = torch.tensor(
        [int(item["flight_id"]) for item in batch], dtype=torch.long
    )
    result["mask"] = torch.ones_like(result["y"], dtype=torch.bool)
    return result


def build_data_bundle(
    data_name: str,
    batch_size: int,
    seed: int,
    cmu_dir: str | Path,
    limit_flights: int | None = None,
    history_seconds: float = 60.0,
    horizon_seconds: float = 30.0,
    stride_seconds: float | None = None,
) -> DataBundle:
    if data_name == "synthetic":
        return build_synthetic_bundle(
            batch_size=batch_size,
            seed=seed,
            history_seconds=history_seconds,
            horizon_seconds=horizon_seconds,
            stride_seconds=stride_seconds,
        )
    if data_name == "cmu":
        return build_cmu_bundle(
            data_dir=cmu_dir,
            batch_size=batch_size,
            seed=seed,
            limit_flights=limit_flights,
            history_seconds=history_seconds,
            horizon_seconds=horizon_seconds,
            stride_seconds=stride_seconds,
        )
    raise ValueError(f"Unsupported dataset: {data_name}")


def build_synthetic_bundle(
    batch_size: int = 16,
    seed: int = 42,
    history_seconds: float = 20.0,
    horizon_seconds: float = 10.0,
    stride_seconds: float | None = None,
) -> DataBundle:
    del stride_seconds
    step_seconds = 1.0
    history_steps = max(2, int(round(history_seconds / step_seconds)))
    horizon_steps = max(1, int(round(horizon_seconds / step_seconds)))
    rng = np.random.default_rng(seed)
    raw_windows: list[dict[str, Any]] = []
    split_sizes = {"train": 96, "val": 24, "test": 24}
    history_names = [
        "obs_x",
        "obs_y",
        "action_x",
        "action_y",
        "wind",
        "battery",
        "power",
    ]
    future_names = ["eta_s", "action_x", "action_y", "forecast_wind", "dt_s"]

    flight_id = 0
    for split, count in split_sizes.items():
        for _ in range(count):
            flight_id += 1
            total_steps = history_steps + horizon_steps
            state = rng.normal(0.0, 0.4, size=2)
            battery = rng.uniform(0.75, 1.0)
            history = []
            future = []
            targets = []
            priors = []
            for step in range(total_steps):
                action = np.array(
                    [math.sin(step / 7.0), math.cos(step / 11.0)], dtype=float
                ) + rng.normal(0.0, 0.08, size=2)
                wind = 0.7 * math.sin(step / 9.0) + rng.normal(0.0, 0.12)
                state = np.array(
                    [
                        0.88 * state[0] + 0.20 * action[0] + 0.10 * wind,
                        0.78 * state[1] + 0.18 * action[1] - 0.08 * wind,
                    ]
                )
                physics_power = (
                    2.0
                    + 0.55 * float(np.dot(action, action))
                    + 0.35 * wind**2
                    + 0.20 * abs(state[1])
                )
                power = (
                    physics_power
                    + 0.25 * math.sin(state[0] * 2.0)
                    + rng.normal(0.0, 0.05)
                )
                battery = max(battery - power / 3600.0, 0.0)
                observation = state + rng.normal(0.0, 0.04, size=2)
                if step < history_steps:
                    history.append([*observation, *action, wind, battery, power])
                else:
                    lead = step - history_steps + 1
                    future.append([lead, *action, wind, 1.0])
                    targets.append(power)
                    priors.append(physics_power)

            raw_windows.append(
                {
                    "split": split,
                    "history": np.asarray(history, dtype=float),
                    "future": np.asarray(future, dtype=float),
                    "y": np.asarray(targets, dtype=float),
                    "physics": np.asarray(priors, dtype=float),
                    "dt": np.ones(horizon_steps, dtype=float),
                    "lead_time": np.arange(1, horizon_steps + 1, dtype=float),
                    "flight_id": flight_id,
                    "target_segment_index": np.arange(horizon_steps, dtype=int),
                }
            )

    return _finalize_bundle(
        raw_windows=raw_windows,
        history_feature_names=history_names,
        future_feature_names=future_names,
        batch_size=batch_size,
        seed=seed,
        description="Causal synthetic history-to-future rollout windows.",
        step_seconds=step_seconds,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        physics_coefficients=None,
    )


def build_cmu_bundle(
    data_dir: str | Path = "data/processed/world_model_cmu_5s",
    batch_size: int = 8,
    seed: int = 42,
    limit_flights: int | None = None,
    history_seconds: float = 60.0,
    horizon_seconds: float = 30.0,
    stride_seconds: float | None = None,
) -> DataBundle:
    root = Path(data_dir)
    segment_path = root / "wm_segments.csv"
    split_path = root / "wm_split.csv"
    config_path = root / "preprocess_config.json"
    if not segment_path.exists() or not split_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Expected wm_segments.csv, wm_split.csv, and preprocess_config.json in {root}. "
            "Run `python -m world_model.preprocess_cmu --window 5s` first."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    step_seconds = float(config["window_seconds"])
    history_steps = max(2, int(round(history_seconds / step_seconds)))
    horizon_steps = max(1, int(round(horizon_seconds / step_seconds)))
    stride_steps = max(
        1,
        int(
            round(
                (stride_seconds if stride_seconds is not None else step_seconds)
                / step_seconds
            )
        ),
    )

    split_by_flight: dict[int, str] = {}
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split_by_flight[int(float(row["flight"]))] = row["split"]

    selected_flights = sorted(split_by_flight)
    if limit_flights is not None:
        selected_flights = _proportional_flight_limit(
            selected_flights, split_by_flight, int(limit_flights)
        )
    selected_set = set(selected_flights)

    all_rows_by_flight: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with segment_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            flight = int(float(row["flight"]))
            if flight in selected_set:
                all_rows_by_flight[flight].append(row)

    raw_windows: list[dict[str, Any]] = []
    for flight in selected_flights:
        rows = sorted(
            all_rows_by_flight[flight], key=lambda row: int(row["segment_index"])
        )
        prepared = _prepare_flight_rows(rows)
        usable = [row for row in prepared if _truthy(row.get("usable_segment"))]
        for run in _contiguous_runs(usable):
            required = history_steps + horizon_steps
            if len(run) < required:
                continue
            for origin in range(
                history_steps, len(run) - horizon_steps + 1, stride_steps
            ):
                history_rows = run[origin - history_steps : origin]
                future_rows = run[origin : origin + horizon_steps]
                history = np.asarray(
                    [
                        [_history_value(row, name) for name in HISTORY_FEATURES]
                        for row in history_rows
                    ],
                    dtype=float,
                )
                future = _future_features(history_rows[-1], future_rows)
                target = np.asarray(
                    [_number(row.get("mean_power_W")) for row in future_rows],
                    dtype=float,
                )
                dt = np.asarray(
                    [
                        _number(row.get("integration_duration_s"))
                        for row in future_rows
                    ],
                    dtype=float,
                )
                lead_time = np.asarray(
                    [
                        _number(row.get("start_time_s"))
                        - _number(history_rows[-1].get("end_time_s"))
                        for row in future_rows
                    ],
                    dtype=float,
                )
                raw_windows.append(
                    {
                        "split": split_by_flight[flight],
                        "history": history,
                        "future": future,
                        "y": target,
                        "dt": dt,
                        "lead_time": lead_time,
                        "flight_id": flight,
                        "target_segment_index": np.asarray(
                            [int(row["segment_index"]) for row in future_rows],
                            dtype=int,
                        ),
                    }
                )

    _require_all_splits(raw_windows)
    train_future = np.concatenate(
        [window["future"] for window in raw_windows if window["split"] == "train"]
    )
    train_target = np.concatenate(
        [window["y"] for window in raw_windows if window["split"] == "train"]
    )
    future_medians = _finite_medians(train_future)
    clean_train_future = _impute(train_future, future_medians)
    physics_coefficients = _fit_physics_prior(clean_train_future, train_target)
    for window in raw_windows:
        clean_future = _impute(window["future"], future_medians)
        window["future"] = clean_future
        window["physics"] = np.maximum(
            _physics_basis(clean_future) @ physics_coefficients, 1.0
        )

    return _finalize_bundle(
        raw_windows=raw_windows,
        history_feature_names=HISTORY_FEATURES,
        future_feature_names=FUTURE_FEATURES,
        batch_size=batch_size,
        seed=seed,
        description=(
            f"CMU causal windows: {history_seconds:g}s history to "
            f"{horizon_seconds:g}s future trajectory from {len(selected_flights)} flights."
        ),
        step_seconds=step_seconds,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        physics_coefficients=physics_coefficients,
    )


def _prepare_flight_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    first_lon = _number(rows[0].get("mean_longitude_deg"))
    first_lat = _number(rows[0].get("mean_latitude_deg"))
    first_alt = _number(rows[0].get("mean_altitude_m"))
    cumulative_energy = 0.0
    prepared = []
    for source in rows:
        row = dict(source)
        lon = _number(row.get("mean_longitude_deg"))
        lat = _number(row.get("mean_latitude_deg"))
        east, north = _local_position(first_lon, first_lat, lon, lat)
        row["local_east_m"] = east
        row["local_north_m"] = north
        altitude = _number(row.get("mean_altitude_m"))
        row["relative_altitude_m"] = (
            altitude - first_alt
            if altitude is not None and first_alt is not None
            else float("nan")
        )
        energy = _number(row.get("segment_energy_Wh"))
        if energy is not None:
            cumulative_energy += energy
        row["cumulative_energy_Wh"] = cumulative_energy
        prepared.append(row)
    return prepared


def _future_features(
    last_history: dict[str, Any], future_rows: list[dict[str, Any]]
) -> np.ndarray:
    wind_east = _number(last_history.get("wind_east_mps")) or 0.0
    wind_north = _number(last_history.get("wind_north_mps")) or 0.0
    wind_speed = math.hypot(wind_east, wind_north)
    origin_end = _number(last_history.get("end_time_s")) or 0.0
    previous = last_history
    result = []
    for row in future_rows:
        previous_time = _segment_center_time(previous)
        current_time = _segment_center_time(row)
        elapsed = max(current_time - previous_time, 1e-3)
        delta_east = _numeric_difference(row, previous, "local_east_m")
        delta_north = _numeric_difference(row, previous, "local_north_m")
        delta_altitude = _numeric_difference(row, previous, "relative_altitude_m")
        horizontal_distance = math.hypot(delta_east, delta_north)
        if horizontal_distance > 1e-6:
            heading_east = delta_east / horizontal_distance
            heading_north = delta_north / horizontal_distance
        else:
            heading_east = _number(previous.get("route_heading_east")) or 0.0
            heading_north = _number(previous.get("route_heading_north")) or 0.0
        wind_along = wind_east * heading_east + wind_north * heading_north
        wind_cross = -wind_east * heading_north + wind_north * heading_east
        start_time = _number(row.get("start_time_s"))
        values = {
            "eta_s": (start_time if start_time is not None else origin_end) - origin_end,
            "planned_duration_s": elapsed,
            "local_east_m": _number(row.get("local_east_m")),
            "local_north_m": _number(row.get("local_north_m")),
            "relative_altitude_m": _number(row.get("relative_altitude_m")),
            "route_delta_east_m": delta_east,
            "route_delta_north_m": delta_north,
            "route_delta_altitude_m": delta_altitude,
            "planned_ground_speed_mps": horizontal_distance / elapsed,
            "planned_vertical_speed_mps": delta_altitude / elapsed,
            "route_heading_east": heading_east,
            "route_heading_north": heading_north,
            "programmed_speed_mps": _number(row.get("programmed_speed_mps")),
            "payload_g": _number(row.get("payload_g")),
            "forecast_wind_speed_mps": wind_speed,
            "forecast_wind_east_mps": wind_east,
            "forecast_wind_north_mps": wind_north,
            "forecast_wind_along_route_mps": wind_along,
            "forecast_wind_cross_route_mps": wind_cross,
        }
        result.append([values[name] for name in FUTURE_FEATURES])
        previous = row
    return np.asarray(result, dtype=float)


def _finalize_bundle(
    raw_windows: list[dict[str, Any]],
    history_feature_names: list[str],
    future_feature_names: list[str],
    batch_size: int,
    seed: int,
    description: str,
    step_seconds: float,
    history_steps: int,
    horizon_steps: int,
    physics_coefficients: np.ndarray | None,
) -> DataBundle:
    _require_all_splits(raw_windows)
    train_history = np.concatenate(
        [window["history"] for window in raw_windows if window["split"] == "train"]
    )
    train_future = np.concatenate(
        [window["future"] for window in raw_windows if window["split"] == "train"]
    )
    train_target = np.concatenate(
        [window["y"] for window in raw_windows if window["split"] == "train"]
    )
    history_medians = _finite_medians(train_history)
    future_medians = _finite_medians(train_future)
    train_history = _impute(train_history, history_medians)
    train_future = _impute(train_future, future_medians)
    history_mean, history_std = _mean_std(train_history)
    future_mean, future_std = _mean_std(train_future)
    target_mean = float(np.mean(train_target))
    target_std = float(np.std(train_target))
    if target_std < 1e-6:
        target_std = 1.0

    finalized: dict[str, list[dict[str, np.ndarray | int]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for window in raw_windows:
        history = _impute(window["history"], history_medians)
        future = _impute(window["future"], future_medians)
        finalized[window["split"]].append(
            {
                "history": ((history - history_mean) / history_std).astype(np.float32),
                "future": ((future - future_mean) / future_std).astype(np.float32),
                "y": ((window["y"] - target_mean) / target_std).astype(np.float32),
                "physics": ((window["physics"] - target_mean) / target_std).astype(
                    np.float32
                ),
                "dt": np.asarray(window["dt"], dtype=np.float32),
                "lead_time": np.asarray(window["lead_time"], dtype=np.float32),
                "flight_id": int(window["flight_id"]),
                "target_segment_index": np.asarray(
                    window["target_segment_index"], dtype=np.int64
                ),
            }
        )

    generator = torch.Generator().manual_seed(seed)

    def loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            WindowDataset(finalized[split]),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            collate_fn=collate_windows,
        )

    preprocessing_state = {
        "history_medians": history_medians.tolist(),
        "history_mean": history_mean.tolist(),
        "history_std": history_std.tolist(),
        "future_medians": future_medians.tolist(),
        "future_mean": future_mean.tolist(),
        "future_std": future_std.tolist(),
        "target_mean": target_mean,
        "target_std": target_std,
        "physics_coefficients": (
            physics_coefficients.tolist() if physics_coefficients is not None else None
        ),
    }
    return DataBundle(
        train_loader=loader("train", True),
        val_loader=loader("val", False),
        test_loader=loader("test", False),
        history_dim=len(history_feature_names),
        future_dim=len(future_feature_names),
        target_mean=target_mean,
        target_std=target_std,
        history_feature_names=history_feature_names,
        future_feature_names=future_feature_names,
        description=description,
        step_seconds=step_seconds,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        preprocessing_state=preprocessing_state,
    )


def _fit_physics_prior(future: np.ndarray, target: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(_physics_basis(future), target, rcond=None)
    return coefficients


def _physics_basis(future: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(FUTURE_FEATURES)}
    ground_speed = future[:, index["planned_ground_speed_mps"]]
    vertical_speed = future[:, index["planned_vertical_speed_mps"]]
    payload_kg = future[:, index["payload_g"]] / 1000.0
    wind_along = future[:, index["forecast_wind_along_route_mps"]]
    wind_cross = future[:, index["forecast_wind_cross_route_mps"]]
    mass = 2.4 + np.maximum(payload_kg, 0.0)
    relative_airspeed = np.sqrt((ground_speed - wind_along) ** 2 + wind_cross**2)
    return np.column_stack(
        [
            np.ones(len(future)),
            mass**1.5,
            relative_airspeed**2,
            relative_airspeed**3,
            mass * 9.81 * np.maximum(vertical_speed, 0.0),
            wind_cross**2,
        ]
    )


def _contiguous_runs(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_index: int | None = None
    for row in rows:
        index = int(row["segment_index"])
        if previous_index is None or index == previous_index + 1:
            current.append(row)
        else:
            if current:
                runs.append(current)
            current = [row]
        previous_index = index
    if current:
        runs.append(current)
    return runs


def _history_value(row: dict[str, Any], name: str) -> float:
    value = _number(row.get(name))
    return value if value is not None else float("nan")


def _numeric_difference(
    current: dict[str, Any], previous: dict[str, Any], name: str
) -> float:
    current_value = _number(current.get(name))
    previous_value = _number(previous.get(name))
    if current_value is None or previous_value is None:
        return 0.0
    return current_value - previous_value


def _segment_center_time(row: dict[str, Any]) -> float:
    start = _number(row.get("start_time_s")) or 0.0
    end = _number(row.get("end_time_s")) or start
    return 0.5 * (start + end)


def _local_position(
    origin_lon: float | None,
    origin_lat: float | None,
    lon: float | None,
    lat: float | None,
) -> tuple[float, float]:
    if None in (origin_lon, origin_lat, lon, lat):
        return float("nan"), float("nan")
    origin_lat_rad = math.radians(float(origin_lat))
    east = (
        EARTH_RADIUS_M
        * math.radians(float(lon) - float(origin_lon))
        * math.cos(origin_lat_rad)
    )
    north = EARTH_RADIUS_M * math.radians(float(lat) - float(origin_lat))
    return east, north


def _finite_medians(values: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(values, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean, np.where(std < 1e-6, 1.0, std)


def _impute(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    row_indices, column_indices = np.where(~np.isfinite(result))
    result[row_indices, column_indices] = medians[column_indices]
    return result


def _require_all_splits(windows: list[dict[str, Any]]) -> None:
    present = {window["split"] for window in windows}
    missing = {"train", "val", "test"} - present
    if missing:
        raise ValueError(f"No windows were generated for splits: {sorted(missing)}")


def _proportional_flight_limit(
    flights: list[int], split_by_flight: dict[int, str], limit: int
) -> list[int]:
    if limit < 3:
        raise ValueError("limit_flights must be at least 3")
    if limit >= len(flights):
        return flights
    by_split: dict[str, list[int]] = defaultdict(list)
    for flight in flights:
        by_split[split_by_flight[flight]].append(flight)
    requested = {
        "train": max(1, round(limit * 0.60)),
        "val": max(1, round(limit * 0.20)),
    }
    requested["test"] = max(1, limit - requested["train"] - requested["val"])
    while sum(requested.values()) > limit:
        requested["train"] -= 1
    selected = []
    for split in ("train", "val", "test"):
        selected.extend(by_split[split][: requested[split]])
    return sorted(selected)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
