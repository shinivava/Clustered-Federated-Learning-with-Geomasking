
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import t, ttest_rel
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

EARTH_RADIUS_KM = 6371.0088
MASKS = (
    "random_perturbation",
    "donut",
    "grid",
    "adaptive_areal",
    "planar_laplace",
)

# Daily file and country-metadata name reconciliation. Extend only after documenting
# the raw data version and the explicit matching decision.
ALIASES = {
    "united states of america": "united states",
    "czechia": "czech republic",
    "north macedonia": "macedonia",
    "republic of moldova": "moldova",
    "timor leste": "east timor",
    "cape verde": "cabo verde",
    "eswatini": "swaziland",
    "myanmar": "burma",
    "brunei darussalam": "brunei",
    "viet nam": "vietnam",
    "syrian arab republic": "syria",
    "russian federation": "russia",
    "iran islamic republic of": "iran",
    "bolivia plurinational state of": "bolivia",
    "venezuela bolivarian republic of": "venezuela",
    "tanzania united republic of": "tanzania",
    "lao peoples democratic republic": "laos",
    "occupied palestinian territory including east jerusalem": "palestine",
}

# Twenty temporal/epidemiological features use present or past observations only.
DYNAMIC_FEATURES = (
    "incidence_14d",
    "cases_per_100k",
    "deaths_per_100k",
    "log_cases_per_100k",
    "log_deaths_per_100k",
    "incidence_lag_1",
    "incidence_lag_2",
    "incidence_lag_3",
    "incidence_lag_7",
    "incidence_delta_1",
    "incidence_delta_7",
    "cases_per_100k_lag_1",
    "incidence_roll_mean_7",
    "incidence_roll_std_7",
    "cases_roll_mean_7",
    "deaths_roll_mean_7",
    "day_sin",
    "day_cos",
    "week_sin",
    "week_cos",
)
# Fifteen static metadata columns make 35 candidates before SelectKBest.
STATIC_FEATURES = (
    "Flu_pneumonia_death_rate_per_100000",
    "h1n1_Cases_confirmed",
    "h1n1_Deaths_confirmed",
    "jan_temp",
    "apr_temp",
    "annual_temp",
    "Jan_precip",
    "Apr_precip",
    "Annual_precip",
    "airport_traffic_2018_thousands",
    "Health_Care_Index",
    "Density_KM2m",
    "Median_age",
    "Population_2020",
    "Fertility_rate",
)


@dataclass(frozen=True)
class ProtocolConfig:
    """All values used by the documented experiment protocol."""

    rounds: int = 25
    local_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_batch_size: int = 64
    eval_batch_size: int = 128
    n_clusters: int = 3
    fedprox_mu: float = 0.01
    label_skew_alpha: float = 0.30
    candidate_features: int = 35
    selected_features: int = 25
    primary_mask: str = "planar_laplace"
    primary_mask_scale_km: float = 1.0
    mask_scales_km: tuple[float, ...] = (0.25, 0.50, 1.00, 2.00)
    reid_threshold_km: float = 0.10
    seeds: tuple[int, ...] = tuple(range(1, 16))
    client_counts: tuple[int, ...] = (8, 10, 15)


@dataclass
class RawData:
    frame: pd.DataFrame
    candidate_features: list[str]
    metadata: dict[str, Any]


@dataclass
class ClientData:
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    @property
    def train_size(self) -> int:
        return int(self.train_y.numel())


@dataclass
class ExperimentData:
    clients: dict[str, ClientData]
    client_names: list[str]
    coordinates: np.ndarray
    densities: np.ndarray
    client_sizes: np.ndarray
    selected_features: list[str]
    partition: str
    seed: int
    metadata: dict[str, Any]


@dataclass
class ModelResult:
    states: dict[str, dict[str, torch.Tensor]]
    client_to_state: dict[str, str]
    validation_history: list[dict[str, float]]


def _normalise_name(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", " ", value.lower().replace("_", " "))
    return re.sub(r"\s+", " ", value).strip()


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    """Use a CPU deterministic setup suitable for repeated seeded experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def destination_points(lat: np.ndarray, lon: np.ndarray, distance_km: np.ndarray, bearing: np.ndarray) -> np.ndarray:
    phi1, lam1 = np.radians(lat), np.radians(lon)
    angular = distance_km / EARTH_RADIUS_KM
    phi2 = np.arcsin(np.sin(phi1) * np.cos(angular) + np.cos(phi1) * np.sin(angular) * np.cos(bearing))
    lam2 = lam1 + np.arctan2(
        np.sin(bearing) * np.sin(angular) * np.cos(phi1),
        np.cos(angular) - np.sin(phi1) * np.sin(phi2),
    )
    return np.column_stack((np.degrees(phi2), (np.degrees(lam2) + 540.0) % 360.0 - 180.0))


def _file_schema_error(path: Path, missing: set[str]) -> ValueError:
    return ValueError(f"{path.name} does not contain required columns: {sorted(missing)}")


def load_raw_data(daily_csv: str | Path, static_csv: str | Path) -> RawData:
    """Load documented source inputs and construct an honest next-day prediction task.

    The target is one iff next-day 14-day incidence is above current 14-day incidence.
    Target-defining future information is not a feature. Each source country is a potential
    simulated client; it is not asserted to be an independently collected organisation.
    """
    daily_path, static_path = Path(daily_csv), Path(static_csv)
    daily, static = pd.read_csv(daily_path), pd.read_csv(static_path)
    daily_required = {
        "dateRep", "cases", "deaths", "countriesAndTerritories", "popData2019",
        "Cumulative_number_for_14_days_of_COVID-19_cases_per_100000",
    }
    static_required = {"country", "latitude", "longitude"}
    if missing := daily_required.difference(daily.columns):
        raise _file_schema_error(daily_path, missing)
    if missing := static_required.difference(static.columns):
        raise _file_schema_error(static_path, missing)

    daily = daily.copy()
    static = static.copy()
    daily["date"] = pd.to_datetime(daily["dateRep"], format="%d/%m/%Y", errors="coerce")
    daily["country_key"] = daily["countriesAndTerritories"].map(_normalise_name).replace(ALIASES)
    static["country_key"] = static["country"].map(_normalise_name)
    if static["country_key"].duplicated().any():
        duplicates = static.loc[static["country_key"].duplicated(keep=False), "country_key"].unique().tolist()
        raise ValueError(f"Duplicate normalized static country keys: {duplicates[:10]}")

    static_columns = ["country_key", "country", "latitude", "longitude"] + [
        c for c in STATIC_FEATURES if c in static.columns
    ]
    merged = daily.merge(static[static_columns], on="country_key", how="inner", validate="many_to_one")
    unmatched = sorted(set(daily["country_key"].dropna()) - set(static["country_key"].dropna()))
    numeric_columns = [
        "cases", "deaths", "popData2019", "Cumulative_number_for_14_days_of_COVID-19_cases_per_100000",
        "latitude", "longitude",
    ] + [c for c in STATIC_FEATURES if c in merged.columns]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(subset=["date", "latitude", "longitude", "popData2019", "Cumulative_number_for_14_days_of_COVID-19_cases_per_100000"])
    merged = merged[merged["latitude"].between(-89.9, 89.9) & merged["longitude"].between(-180, 180)].copy()
    merged["cases"] = merged["cases"].clip(lower=0)
    merged["deaths"] = merged["deaths"].clip(lower=0)
    merged["incidence_14d"] = merged["Cumulative_number_for_14_days_of_COVID-19_cases_per_100000"]
    merged["cases_per_100k"] = 1e5 * merged["cases"] / merged["popData2019"].clip(lower=1.0)
    merged["deaths_per_100k"] = 1e5 * merged["deaths"] / merged["popData2019"].clip(lower=1.0)
    merged = merged.sort_values(["country_key", "date"]).reset_index(drop=True)

    def build_country_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["log_cases_per_100k"] = np.log1p(group["cases_per_100k"])
        group["log_deaths_per_100k"] = np.log1p(group["deaths_per_100k"])
        for lag in (1, 2, 3, 7):
            group[f"incidence_lag_{lag}"] = group["incidence_14d"].shift(lag)
        group["incidence_delta_1"] = group["incidence_14d"] - group["incidence_14d"].shift(1)
        group["incidence_delta_7"] = group["incidence_14d"] - group["incidence_14d"].shift(7)
        group["cases_per_100k_lag_1"] = group["cases_per_100k"].shift(1)
        group["incidence_roll_mean_7"] = group["incidence_14d"].rolling(7, min_periods=1).mean()
        group["incidence_roll_std_7"] = group["incidence_14d"].rolling(7, min_periods=2).std()
        group["cases_roll_mean_7"] = group["cases_per_100k"].rolling(7, min_periods=1).mean()
        group["deaths_roll_mean_7"] = group["deaths_per_100k"].rolling(7, min_periods=1).mean()
        group["future_incidence_14d"] = group["incidence_14d"].shift(-1)
        return group

    merged = pd.concat([build_country_features(g) for _, g in merged.groupby("country_key", sort=True)], ignore_index=True)
    merged["target"] = np.where(
        merged["future_incidence_14d"].notna(),
        (merged["future_incidence_14d"] > merged["incidence_14d"]).astype(int),
        np.nan,
    )
    merged = merged.dropna(subset=["target"]).copy()
    merged["target"] = merged["target"].astype(int)
    merged["day_of_year"] = merged["date"].dt.dayofyear.astype(float)
    merged["day_sin"] = np.sin(2.0 * np.pi * merged["day_of_year"] / 366.0)
    merged["day_cos"] = np.cos(2.0 * np.pi * merged["day_of_year"] / 366.0)
    week = merged["date"].dt.isocalendar().week.astype(float)
    merged["week_sin"] = np.sin(2.0 * np.pi * week / 53.0)
    merged["week_cos"] = np.cos(2.0 * np.pi * week / 53.0)

    split_frames: list[pd.DataFrame] = []
    for name, group in merged.groupby("country_key", sort=True):
        group = group.sort_values("date").copy()
        n = len(group)
        train_end, val_end = int(math.floor(0.80 * n)), int(math.floor(0.90 * n))
        group["split"] = "test"
        group.iloc[:train_end, group.columns.get_loc("split")] = "train"
        group.iloc[train_end:val_end, group.columns.get_loc("split")] = "val"
        split_frames.append(group)
    merged = pd.concat(split_frames, ignore_index=True)

    split_counts = merged.groupby(["country_key", "split"]).size().unstack(fill_value=0)
    eligible = split_counts.index[(split_counts.get("train", 0) >= 32) & (split_counts.get("val", 0) >= 8) & (split_counts.get("test", 0) >= 8)]
    merged = merged[merged["country_key"].isin(eligible)].copy().reset_index(drop=True)
    merged["row_id"] = np.arange(len(merged), dtype=int)

    candidates = [c for c in DYNAMIC_FEATURES + STATIC_FEATURES if c in merged.columns]
    train = merged[merged["split"] == "train"]
    candidates = [c for c in candidates if train[c].notna().any()]
    metadata = {
        "daily_csv": str(daily_path),
        "daily_sha256": sha256sum(daily_path),
        "static_csv": str(static_path),
        "static_sha256": sha256sum(static_path),
        "rows_after_join_and_eligibility": int(len(merged)),
        "eligible_source_clients": int(merged["country_key"].nunique()),
        "candidate_feature_count": int(len(candidates)),
        "candidate_features": candidates,
        "class_balance_positive": float(merged["target"].mean()),
        "prediction_target": "Next-day 14-day incidence per 100,000 is greater than current 14-day incidence per 100,000.",
        "source_client_definition": "Country-level simulated client derived from an aligned country time series.",
        "unmatched_daily_country_keys": unmatched,
    }
    return RawData(frame=merged, candidate_features=candidates, metadata=metadata)


def _select_source_clients(raw: RawData, n_clients: int, seed: int) -> list[str]:
    # Restrict the randomly sampled cohort to countries that provide every declared
    # static feature. This avoids a seed silently losing a static feature merely
    # because all selected countries lack that metadata field.
    static = [column for column in raw.candidate_features if column in STATIC_FEATURES]
    coverage = raw.frame.groupby("country_key", sort=True)[static].agg(lambda series: series.notna().any())
    available = sorted(coverage.index[coverage.all(axis=1)].tolist())
    if n_clients > len(available):
        raise ValueError(
            f"Requested {n_clients} clients but only {len(available)} source clients have complete declared static metadata."
        )
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(available, size=n_clients, replace=False).tolist())


def _assign_iid(frame: pd.DataFrame, clients: list[str], rng: np.random.Generator) -> np.ndarray:
    assigned = np.empty(len(frame), dtype=object)
    for split in ("train", "val", "test"):
        split_mask = frame["split"].to_numpy() == split
        for label in (0, 1):
            ids = np.flatnonzero(split_mask & (frame["target"].to_numpy() == label))
            rng.shuffle(ids)
            for client, block in zip(clients, np.array_split(ids, len(clients))):
                assigned[block] = client
    return assigned


def _assign_label_skew(frame: pd.DataFrame, clients: list[str], rng: np.random.Generator, alpha: float) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("label_skew_alpha must be positive.")
    assigned = np.empty(len(frame), dtype=object)
    label_preferences = {label: rng.dirichlet(np.full(len(clients), alpha)) for label in (0, 1)}
    targets = frame["target"].to_numpy()
    splits = frame["split"].to_numpy()
    for split in ("train", "val", "test"):
        for label in (0, 1):
            ids = np.flatnonzero((splits == split) & (targets == label))
            rng.shuffle(ids)
            # Reserve one observation from each class for every client in every
            # split. The remaining records follow the Dirichlet label-skew draw.
            # This preserves a non-IID allocation while preventing empty
            # validation/test partitions and singleton BatchNorm batches.
            if len(ids) < len(clients):
                raise ValueError(
                    f"Insufficient {split} records of label {label} to construct a label-skew split with {len(clients)} clients."
                )
            counts = 1 + rng.multinomial(len(ids) - len(clients), label_preferences[label])
            start = 0
            for client, count in zip(clients, counts):
                assigned[ids[start:start + count]] = client
                start += count
    return assigned


def prepare_experiment_data(raw: RawData, n_clients: int, partition: str, seed: int, config: ProtocolConfig) -> ExperimentData:
    """Build a seed-specific client cohort, partitions, and train-only feature pipeline."""
    if partition not in {"geographic", "iid", "label_skew"}:
        raise ValueError("partition must be geographic, iid, or label_skew")
    selected_sources = _select_source_clients(raw, n_clients, seed)
    frame = raw.frame[raw.frame["country_key"].isin(selected_sources)].copy().reset_index(drop=True)
    train = frame[frame["split"] == "train"]
    available = [c for c in raw.candidate_features if train[c].notna().any()]
    if len(available) < config.selected_features:
        raise ValueError(f"Only {len(available)} usable train-fold features; need {config.selected_features}.")
    if len(available) != config.candidate_features:
        print(f"WARNING: {len(available)} usable candidates, not the configured {config.candidate_features}. Report the measured count.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_x = imputer.fit_transform(train[available])
    train_x = scaler.fit_transform(train_x)
    selector = SelectKBest(score_func=f_classif, k=config.selected_features)
    selector.fit(train_x, train["target"].to_numpy())
    all_x = scaler.transform(imputer.transform(frame[available]))
    x = selector.transform(all_x).astype(np.float32)
    selected_features = np.asarray(available)[selector.get_support()].tolist()
    frame["feature_row"] = np.arange(len(frame), dtype=int)

    rng = np.random.default_rng(seed + 991)
    client_names = selected_sources[:] if partition == "geographic" else [f"simulated_client_{i + 1:02d}" for i in range(n_clients)]
    if partition == "geographic":
        assignments = frame["country_key"].to_numpy(dtype=object)
    elif partition == "iid":
        assignments = _assign_iid(frame, client_names, rng)
    else:
        assignments = _assign_label_skew(frame, client_names, rng, config.label_skew_alpha)
    if np.any(pd.isna(assignments)):
        raise RuntimeError("Partition left records unassigned.")
    frame["client_id"] = assignments

    # Coordinates and density are client metadata, never predictive features.
    source_meta = frame.groupby("country_key", sort=True).first()[["latitude", "longitude", "Density_KM2m"]]
    if partition == "geographic":
        meta_sources = {name: name for name in client_names}
    else:
        meta_sources = dict(zip(client_names, selected_sources))
    clients: dict[str, ClientData] = {}
    coordinates, densities, label_rows = [], [], []
    for client in client_names:
        current = frame[frame["client_id"] == client]
        tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for split in ("train", "val", "test"):
            subset = current[current["split"] == split]
            if len(subset) < 2:
                raise ValueError(f"{partition} partition gave {client} fewer than two {split} records; use another seed or alpha.")
            positions = subset["feature_row"].to_numpy(dtype=int)
            tensors[split] = (
                torch.tensor(x[positions], dtype=torch.float32),
                torch.tensor(subset["target"].to_numpy(dtype=np.float32), dtype=torch.float32),
            )
        clients[client] = ClientData(client, tensors["train"][0], tensors["train"][1], tensors["val"][0], tensors["val"][1], tensors["test"][0], tensors["test"][1])
        source = meta_sources[client]
        coordinates.append((float(source_meta.loc[source, "latitude"]), float(source_meta.loc[source, "longitude"])))
        density = pd.to_numeric(pd.Series([source_meta.loc[source, "Density_KM2m"]]), errors="coerce").iloc[0]
        densities.append(float(density) if pd.notna(density) and density > 0 else np.nan)
        label_rows.append({
            "client_id": client,
            "source_coordinate_country": source,
            "train_n": clients[client].train_size,
            "val_n": int(clients[client].val_y.numel()),
            "test_n": int(clients[client].test_y.numel()),
            "train_positive_rate": float(clients[client].train_y.mean().item()),
        })
    metadata = {
        "seed": seed,
        "partition": partition,
        "selected_source_clients": selected_sources,
        "selected_features": selected_features,
        "candidate_features_used": available,
        "feature_selection": "Median imputation, standardization, and SelectKBest(f_classif) fit on pooled selected-client training folds only.",
        "preprocessing_scope": "Centralized evaluation-simulation preprocessing; not a privacy-preserving distributed feature-selection protocol.",
        "client_partition_note": "IID and label_skew are simulated partitions; geographic uses each selected country time series as a simulated client.",
        "client_statistics": label_rows,
    }
    return ExperimentData(
        clients=clients,
        client_names=client_names,
        coordinates=np.asarray(coordinates, dtype=float),
        densities=np.asarray(densities, dtype=float),
        client_sizes=np.asarray([clients[n].train_size for n in client_names], dtype=float),
        selected_features=selected_features,
        partition=partition,
        seed=seed,
        metadata=metadata,
    )


class FourHiddenLayerBNNet(torch.nn.Module):
    """Four hidden dense layers, each followed by batch normalization, ReLU and dropout."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        widths, dropouts = (128, 64, 32, 16), (0.30, 0.20, 0.10, 0.10)
        modules: list[torch.nn.Module] = []
        previous = input_dim
        for width, dropout in zip(widths, dropouts):
            modules.extend((torch.nn.Linear(previous, width), torch.nn.BatchNorm1d(width), torch.nn.ReLU(), torch.nn.Dropout(dropout)))
            previous = width
        modules.append(torch.nn.Linear(previous, 1))
        self.network = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def new_model(input_dim: int) -> FourHiddenLayerBNNet:
    return FourHiddenLayerBNNet(input_dim)


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    if len(y) < 2:
        raise ValueError("Batch-normalized model requires at least two local records.")
    actual_batch = min(batch_size, len(y))
    # Avoid the final singleton batch, which cannot update BatchNorm in train mode.
    drop_last = len(y) > actual_batch and len(y) % actual_batch == 1
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y), batch_size=actual_batch, shuffle=shuffle, drop_last=drop_last, generator=generator)


def _state_average(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("Cannot aggregate an empty set of client states.")
    total = float(sum(weights))
    output: dict[str, torch.Tensor] = {}
    for key in states[0]:
        reference = states[0][key]
        if torch.is_floating_point(reference):
            aggregate = torch.zeros_like(reference)
            for state, weight in zip(states, weights):
                aggregate += state[key] * (float(weight) / total)
            output[key] = aggregate
        else:
            output[key] = reference.detach().clone()
    return output


def _local_train(
    initial_state: dict[str, torch.Tensor],
    client: ClientData,
    input_dim: int,
    config: ProtocolConfig,
    seed: int,
    prox_reference: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    model = new_model(input_dim)
    model.load_state_dict(initial_state)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    reference_parameters = None
    if prox_reference is not None:
        reference_parameters = {name: value.detach() for name, value in new_model(input_dim).named_parameters()}
        reference_parameters = {name: prox_reference[name].detach() for name in reference_parameters}
    losses: list[float] = []
    for epoch in range(config.local_epochs):
        for batch_x, batch_y in _loader(client.train_x, client.train_y, config.train_batch_size, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
            if reference_parameters is not None:
                proximal = sum(torch.sum((parameter - reference_parameters[name]) ** 2) for name, parameter in model.named_parameters())
                loss = loss + 0.5 * config.fedprox_mu * proximal
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
    return clone_state(model.state_dict()), float(np.mean(losses))


def _model_loss(state: dict[str, torch.Tensor], client: ClientData, split: str, input_dim: int, batch_size: int) -> float:
    model = new_model(input_dim)
    model.load_state_dict(state)
    model.eval()
    x, y = (client.train_x, client.train_y) if split == "train" else (client.val_x, client.val_y)
    losses: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y in _loader(x, y, batch_size, False, 123):
            losses.extend(torch.nn.functional.binary_cross_entropy_with_logits(model(batch_x), batch_y, reduction="none").cpu().numpy().tolist())
    return float(np.mean(losses))


def _probabilities(state: dict[str, torch.Tensor], x: torch.Tensor, input_dim: int, batch_size: int) -> np.ndarray:
    model = new_model(input_dim)
    model.load_state_dict(state)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        dummy_y = torch.zeros(len(x), dtype=torch.float32)
        for batch_x, _ in _loader(x, dummy_y, batch_size, False, 321):
            chunks.append(torch.sigmoid(model(batch_x)).cpu().numpy())
    return np.concatenate(chunks)


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(int)
    result = {
        "accuracy": float(100.0 * accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
    }
    try:
        result["auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        result["auc"] = float("nan")
    return result


def evaluate_model(result: ModelResult, data: ExperimentData, split: str, config: ProtocolConfig) -> dict[str, float]:
    probabilities, labels = [], []
    input_dim = len(data.selected_features)
    for name in data.client_names:
        client = data.clients[name]
        x, y = (client.val_x, client.val_y) if split == "val" else (client.test_x, client.test_y)
        probabilities.append(_probabilities(result.states[result.client_to_state[name]], x, input_dim, config.eval_batch_size))
        labels.append(y.cpu().numpy())
    return classification_metrics(np.concatenate(probabilities), np.concatenate(labels))


def run_fedavg(data: ExperimentData, config: ProtocolConfig, seed: int, fedprox: bool = False) -> ModelResult:
    set_seed(seed)
    input_dim = len(data.selected_features)
    state = clone_state(new_model(input_dim).state_dict())
    history: list[dict[str, float]] = []
    for round_id in range(config.rounds):
        updates, losses = [], []
        for index, name in enumerate(data.client_names):
            update, loss = _local_train(state, data.clients[name], input_dim, config, seed + 1000 * round_id + index, state if fedprox else None)
            updates.append(update)
            losses.append(loss)
        state = _state_average(updates, data.client_sizes.tolist())
        temporary = ModelResult({"global": state}, {name: "global" for name in data.client_names}, [])
        history.append({"round": round_id + 1, "mean_local_loss": float(np.mean(losses)), **evaluate_model(temporary, data, "val", config)})
    return ModelResult({"global": state}, {name: "global" for name in data.client_names}, history)


def _run_clustered(data: ExperimentData, assignments: dict[str, str], config: ProtocolConfig, seed: int) -> ModelResult:
    set_seed(seed)
    input_dim = len(data.selected_features)
    initial = clone_state(new_model(input_dim).state_dict())
    keys = sorted(set(assignments.values()))
    states = {key: clone_state(initial) for key in keys}
    history: list[dict[str, float]] = []
    for round_id in range(config.rounds):
        mean_losses = []
        for key in keys:
            names = [name for name in data.client_names if assignments[name] == key]
            updates, weights = [], []
            for position, name in enumerate(names):
                update, loss = _local_train(states[key], data.clients[name], input_dim, config, seed + 10000 * round_id + 101 * position + int(key.split("_")[-1]) if "_" in key else seed + position)
                updates.append(update)
                weights.append(data.clients[name].train_size)
                mean_losses.append(loss)
            states[key] = _state_average(updates, weights)
        temporary = ModelResult(states, assignments, [])
        history.append({"round": round_id + 1, "mean_local_loss": float(np.mean(mean_losses)), **evaluate_model(temporary, data, "val", config)})
    return ModelResult(states, assignments, history)


def run_ifca(data: ExperimentData, config: ProtocolConfig, seed: int) -> ModelResult:
    """IFCA-style alternating train-loss assignments and cluster-model updates."""
    set_seed(seed)
    input_dim = len(data.selected_features)
    initial = clone_state(new_model(input_dim).state_dict())
    keys = [f"ifca_{i}" for i in range(config.n_clusters)]
    states = {key: clone_state(initial) for key in keys}
    assignments = {name: keys[index % len(keys)] for index, name in enumerate(data.client_names)}
    history: list[dict[str, float]] = []
    for round_id in range(config.rounds):
        grouped: dict[str, list[tuple[dict[str, torch.Tensor], int]]] = {key: [] for key in keys}
        losses = []
        for index, name in enumerate(data.client_names):
            client = data.clients[name]
            chosen = min(keys, key=lambda key: _model_loss(states[key], client, "train", input_dim, config.eval_batch_size))
            assignments[name] = chosen
            update, loss = _local_train(states[chosen], client, input_dim, config, seed + 7000 * round_id + index)
            grouped[chosen].append((update, client.train_size))
            losses.append(loss)
        for key, updates in grouped.items():
            if updates:
                states[key] = _state_average([state for state, _ in updates], [weight for _, weight in updates])
        temporary = ModelResult(states, assignments.copy(), [])
        history.append({"round": round_id + 1, "mean_local_loss": float(np.mean(losses)), **evaluate_model(temporary, data, "val", config)})
    # Validation-based final selection prevents reusing test loss for model assignment.
    for name in data.client_names:
        assignments[name] = min(keys, key=lambda key: _model_loss(states[key], data.clients[name], "val", input_dim, config.eval_batch_size))
    return ModelResult(states, assignments, history)


def run_local_only(data: ExperimentData, config: ProtocolConfig, seed: int) -> ModelResult:
    set_seed(seed)
    input_dim = len(data.selected_features)
    initial = clone_state(new_model(input_dim).state_dict())
    states: dict[str, dict[str, torch.Tensor]] = {}
    history = []
    local_config = ProtocolConfig(**{**asdict(config), "local_epochs": config.local_epochs * config.rounds, "rounds": 1})
    for index, name in enumerate(data.client_names):
        state, _ = _local_train(initial, data.clients[name], input_dim, local_config, seed + index)
        states[name] = state
    result = ModelResult(states, {name: name for name in data.client_names}, history)
    return result


def to_unit_sphere(coordinates: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(coordinates[:, 0]), np.radians(coordinates[:, 1])
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def kmeans_assignment(coordinates: np.ndarray, client_sizes: np.ndarray, n_clusters: int, seed: int, weighted: bool) -> np.ndarray:
    if n_clusters > len(coordinates):
        raise ValueError("n_clusters cannot exceed number of clients.")
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    features = to_unit_sphere(coordinates)
    if weighted:
        return kmeans.fit_predict(features, sample_weight=client_sizes)
    return kmeans.fit_predict(features)


def clustered_result(data: ExperimentData, coordinates: np.ndarray, config: ProtocolConfig, seed: int, weighted: bool, random_clusters: bool = False) -> tuple[ModelResult, np.ndarray]:
    if random_clusters:
        rng = np.random.default_rng(seed)
        labels = np.tile(np.arange(config.n_clusters), int(np.ceil(len(data.client_names) / config.n_clusters)))[:len(data.client_names)]
        rng.shuffle(labels)
    else:
        labels = kmeans_assignment(coordinates, data.client_sizes, config.n_clusters, seed, weighted)
    assignments = {name: f"cluster_{int(labels[index])}" for index, name in enumerate(data.client_names)}
    return _run_clustered(data, assignments, config, seed), labels


def mask_coordinates(coordinates: np.ndarray, densities: np.ndarray, method: str, scale_km: float, seed: int) -> np.ndarray:
    """Client-side coordinate masking for clustering metadata.

    `planar_laplace` is a planar-Laplace perturbation parameterized by its expected
    displacement. It is not labelled a formal global geo-indistinguishability guarantee
    because the input may cover a large geographic extent and no planar-CRS proof is made.
    """
    if method not in MASKS:
        raise ValueError(f"Unknown mask: {method}")
    if scale_km <= 0:
        raise ValueError("Mask scale must be positive.")
    rng = np.random.default_rng(seed)
    lat, lon = coordinates[:, 0], coordinates[:, 1]
    bearing = rng.uniform(0, 2 * np.pi, len(coordinates))
    if method == "random_perturbation":
        distance = scale_km * np.sqrt(rng.uniform(0, 1, len(coordinates)))
        return destination_points(lat, lon, distance, bearing)
    if method == "donut":
        distance = scale_km * np.sqrt(rng.uniform(0.25, 1, len(coordinates)))
        return destination_points(lat, lon, distance, bearing)
    if method == "grid":
        # Web-Mercator grid quantization; the scale is a projected-coordinate cell width.
        radius = EARTH_RADIUS_KM
        x = radius * np.radians(lon)
        y = radius * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
        origin = rng.uniform(0, scale_km, size=2)
        x = (np.floor((x - origin[0]) / scale_km) + 0.5) * scale_km + origin[0]
        y = (np.floor((y - origin[1]) / scale_km) + 0.5) * scale_km + origin[1]
        return np.column_stack((np.degrees(2 * np.arctan(np.exp(y / radius)) - np.pi / 2), np.degrees(x / radius)))
    if method == "adaptive_areal":
        positive = densities[np.isfinite(densities) & (densities > 0)]
        median = np.median(np.log(positive)) if len(positive) else 0.0
        normalized = np.zeros(len(densities))
        valid = np.isfinite(densities) & (densities > 0)
        spread = np.std(np.log(positive)) if len(positive) else 1.0
        spread = spread if spread > 1e-8 else 1.0
        normalized[valid] = (median - np.log(densities[valid])) / spread
        multiplier = np.clip(1.0 + 0.35 * normalized, 0.5, 1.5)
        distance = scale_km * multiplier * np.sqrt(rng.uniform(0, 1, len(coordinates)))
        return destination_points(lat, lon, distance, bearing)
    # Planar Laplace radial distribution: E[r] = 2 / epsilon = scale_km.
    epsilon_per_km = 2.0 / scale_km
    distance = rng.gamma(shape=2.0, scale=1.0 / epsilon_per_km, size=len(coordinates))
    return destination_points(lat, lon, distance, bearing)


def privacy_metrics(original: np.ndarray, masked: np.ndarray, unmasked_labels: np.ndarray, masked_labels: np.ndarray, threshold_km: float) -> dict[str, float]:
    """Evaluate a declared nearest-candidate centroid attacker.

    The attacker knows all candidate client centroids and sees a masked centroid. AIE is
    the geodesic error between its inferred candidate and the true centroid. Re-ID@100m
    is the fraction with inference error <= 0.10 km, not raw masking displacement.
    """
    distances = haversine_km(masked[:, 0, None], masked[:, 1, None], original[None, :, 0], original[None, :, 1])
    inferred = distances.argmin(axis=1)
    true_index = np.arange(len(original))
    inference_error = haversine_km(original[:, 0], original[:, 1], original[inferred, 0], original[inferred, 1])
    displacement = haversine_km(original[:, 0], original[:, 1], masked[:, 0], masked[:, 1])
    return {
        "aie_km": float(np.mean(inference_error)),
        "reid_at_100m_percent": float(100.0 * np.mean(inference_error <= threshold_km)),
        "raw_mask_displacement_km": float(np.mean(displacement)),
        "cluster_ari": float(adjusted_rand_score(unmasked_labels, masked_labels)),
    }


def mia_metrics(result: ModelResult, data: ExperimentData, config: ProtocolConfig) -> dict[str, float]:
    """Query-only loss-based MIA summary; AUC is threshold-independent.

    Members are the records used by the target FL training; non-members are its fixed
    held-out test records. This is an evaluation attack, not a formal DP guarantee.
    """
    input_dim = len(data.selected_features)
    scores, labels = [], []
    for name in data.client_names:
        client = data.clients[name]
        state = result.states[result.client_to_state[name]]
        model = new_model(input_dim)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            train_loss = torch.nn.functional.binary_cross_entropy_with_logits(model(client.train_x), client.train_y, reduction="none").cpu().numpy()
            test_loss = torch.nn.functional.binary_cross_entropy_with_logits(model(client.test_x), client.test_y, reduction="none").cpu().numpy()
        scores.extend((-train_loss).tolist())
        labels.extend(np.ones(len(train_loss), dtype=int).tolist())
        scores.extend((-test_loss).tolist())
        labels.extend(np.zeros(len(test_loss), dtype=int).tolist())
    try:
        auc = float(roc_auc_score(np.asarray(labels), np.asarray(scores)))
    except ValueError:
        auc = float("nan")
    return {"mia_loss_auc": auc, "mia_advantage": float(abs(2.0 * auc - 1.0)) if np.isfinite(auc) else float("nan")}


def _record(result: ModelResult, data: ExperimentData, config: ProtocolConfig, seed: int, experiment: str, method: str, n_clients: int, mask: str | None = None, scale_km: float | None = None, privacy: dict[str, float] | None = None) -> dict[str, Any]:
    val = evaluate_model(result, data, "val", config)
    test = evaluate_model(result, data, "test", config)
    return {
        "experiment": experiment,
        "seed": seed,
        "partition": data.partition,
        "n_clients": n_clients,
        "k": config.n_clusters,
        "method": method,
        "mask": mask,
        "mask_scale_km": scale_km,
        "selected_features_json": json.dumps(data.selected_features),
        "source_clients_json": json.dumps(data.metadata["selected_source_clients"]),
        **{f"val_{key}": value for key, value in val.items()},
        **{f"test_{key}": value for key, value in test.items()},
        **mia_metrics(result, data, config),
        **(privacy or {}),
    }


def main_comparison(raw: RawData, output_dir: str | Path, config: ProtocolConfig, partitions: tuple[str, ...] = ("iid", "geographic", "label_skew")) -> pd.DataFrame:
    """FedAvg/FedProx/IFCA + unmasked and masked weighted geographic-clustered FL."""
    rows: list[dict[str, Any]] = []
    output = Path(output_dir)
    for seed in config.seeds:
        for partition in partitions:
            for n_clients in config.client_counts:
                data = prepare_experiment_data(raw, n_clients, partition, seed, config)
                fedavg = run_fedavg(data, config, seed)
                fedprox = run_fedavg(data, config, seed, fedprox=True)
                ifca = run_ifca(data, config, seed)
                unmasked, unmasked_labels = clustered_result(data, data.coordinates, config, seed, weighted=True)
                masked_coords = mask_coordinates(data.coordinates, data.densities, config.primary_mask, config.primary_mask_scale_km, seed + 37)
                proposed, masked_labels = clustered_result(data, masked_coords, config, seed, weighted=True)
                privacy = privacy_metrics(data.coordinates, masked_coords, unmasked_labels, masked_labels, config.reid_threshold_km)
                rows.extend((
                    _record(fedavg, data, config, seed, "main", "FedAvg", n_clients),
                    _record(fedprox, data, config, seed, "main", "FedProx", n_clients),
                    _record(ifca, data, config, seed, "main", "IFCA", n_clients),
                    _record(unmasked, data, config, seed, "main", "WeightedClusteredUnmasked", n_clients, privacy={"aie_km": 0.0, "reid_at_100m_percent": 100.0, "raw_mask_displacement_km": 0.0, "cluster_ari": 1.0}),
                    _record(proposed, data, config, seed, "main", "GeoFedMask", n_clients, config.primary_mask, config.primary_mask_scale_km, privacy),
                ))
                _checkpoint(rows, output / "main_seed_results.csv")
    return pd.DataFrame(rows)


def privacy_utility_experiment(raw: RawData, output_dir: str | Path, config: ProtocolConfig, n_clients: int = 15, partition: str = "geographic") -> pd.DataFrame:
    """Five masks x four prespecified levels under otherwise identical GeoFedMask training."""
    rows: list[dict[str, Any]] = []
    output = Path(output_dir)
    for seed in config.seeds:
        data = prepare_experiment_data(raw, n_clients, partition, seed, config)
        _, unmasked_labels = clustered_result(data, data.coordinates, config, seed, weighted=True)
        for method_index, method in enumerate(MASKS):
            for level, scale in enumerate(config.mask_scales_km, start=1):
                masked = mask_coordinates(data.coordinates, data.densities, method, scale, 100_000 * seed + 100 * method_index + level)
                result, labels = clustered_result(data, masked, config, seed, weighted=True)
                privacy = privacy_metrics(data.coordinates, masked, unmasked_labels, labels, config.reid_threshold_km)
                row = _record(result, data, config, seed, "privacy_utility", "GeoFedMask", n_clients, method, scale, privacy)
                row["mask_level"] = level
                rows.append(row)
        _checkpoint(rows, output / "privacy_utility_seed_results.csv")
    return pd.DataFrame(rows)


def ablation_experiment(raw: RawData, output_dir: str | Path, config: ProtocolConfig, n_clients: int = 15, partition: str = "geographic") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    output = Path(output_dir)
    for seed in config.seeds:
        data = prepare_experiment_data(raw, n_clients, partition, seed, config)
        unmasked, unmasked_labels = clustered_result(data, data.coordinates, config, seed, weighted=True)
        masked = mask_coordinates(data.coordinates, data.densities, config.primary_mask, config.primary_mask_scale_km, seed + 71)
        random_result, _ = clustered_result(data, masked, config, seed, weighted=True, random_clusters=True)
        unweighted_result, unweighted_labels = clustered_result(data, masked, config, seed, weighted=False)
        proposed, weighted_labels = clustered_result(data, masked, config, seed, weighted=True)
        privacy = privacy_metrics(data.coordinates, masked, unmasked_labels, weighted_labels, config.reid_threshold_km)
        rows.extend((
            _record(run_fedavg(data, config, seed), data, config, seed, "ablation", "FedAvg_K1", n_clients),
            _record(random_result, data, config, seed, "ablation", "RandomClusters_K3", n_clients, config.primary_mask, config.primary_mask_scale_km),
            _record(unweighted_result, data, config, seed, "ablation", "UnweightedKMeansMasked", n_clients, config.primary_mask, config.primary_mask_scale_km, privacy_metrics(data.coordinates, masked, unmasked_labels, unweighted_labels, config.reid_threshold_km)),
            _record(proposed, data, config, seed, "ablation", "WeightedKMeansMasked_GeoFedMask", n_clients, config.primary_mask, config.primary_mask_scale_km, privacy),
            _record(unmasked, data, config, seed, "ablation", "WeightedKMeansUnmaskedOracle", n_clients, privacy={"aie_km": 0.0, "reid_at_100m_percent": 100.0, "raw_mask_displacement_km": 0.0, "cluster_ari": 1.0}),
            _record(run_local_only(data, config, seed), data, config, seed, "ablation", "LocalOnly_KN", n_clients),
        ))
        _checkpoint(rows, output / "ablation_seed_results.csv")
    return pd.DataFrame(rows)


def sensitivity_experiment(raw: RawData, output_dir: str | Path, config: ProtocolConfig, client_counts: tuple[int, ...] = (8, 15, 30), cluster_counts: tuple[int, ...] = (2, 3, 5), partition: str = "geographic") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    output = Path(output_dir)
    for seed in config.seeds:
        for n_clients in client_counts:
            for k in cluster_counts:
                sensitivity_config = ProtocolConfig(**{**asdict(config), "n_clusters": k})
                data = prepare_experiment_data(raw, n_clients, partition, seed, sensitivity_config)
                unmasked, unmasked_labels = clustered_result(data, data.coordinates, sensitivity_config, seed, weighted=True)
                masked = mask_coordinates(data.coordinates, data.densities, sensitivity_config.primary_mask, sensitivity_config.primary_mask_scale_km, seed + 91)
                proposed, labels = clustered_result(data, masked, sensitivity_config, seed, weighted=True)
                privacy = privacy_metrics(data.coordinates, masked, unmasked_labels, labels, sensitivity_config.reid_threshold_km)
                row = _record(proposed, data, sensitivity_config, seed, "sensitivity", "GeoFedMask", n_clients, sensitivity_config.primary_mask, sensitivity_config.primary_mask_scale_km, privacy)
                row["k"] = k
                rows.append(row)
        _checkpoint(rows, output / "sensitivity_seed_results.csv")
    return pd.DataFrame(rows)


def _checkpoint(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Mean, SD, and 95% t-CI from per-seed rows; no numerical values are hard-coded."""
    numeric = [c for c in frame.columns if c.startswith(("val_", "test_", "mia_", "aie_", "reid_", "raw_", "cluster_"))]
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, keys))
        record["n_seeds"] = int(group["seed"].nunique())
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci = float(t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = sd
            record[f"{metric}_ci95_halfwidth"] = ci
        records.append(record)
    return pd.DataFrame(records)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    output = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    ordered = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, p[index] * (m - rank)))
        output[index] = running
    return output.tolist()


def paired_tests(frame: pd.DataFrame, reference: str = "GeoFedMask", metric: str = "test_accuracy") -> pd.DataFrame:
    """Matched-seed paired tests with Holm correction within each setting."""
    # Method-specific mask fields must not define the comparison stratum: in the
    # main comparison GeoFedMask alone has a mask label, while each baseline is
    # otherwise evaluated under the same seed, partition, client count, and K.
    setting_columns = [c for c in ("experiment", "partition", "n_clients", "k") if c in frame.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(setting_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ref = group[group["method"] == reference][["seed", metric]].rename(columns={metric: "reference"})
        if ref.empty:
            continue
        local_rows = []
        for method in sorted(set(group["method"]) - {reference}):
            other = group[group["method"] == method][["seed", metric]].rename(columns={metric: "baseline"})
            joined = ref.merge(other, on="seed", how="inner").dropna()
            diff = (joined["reference"] - joined["baseline"]).to_numpy(dtype=float)
            if len(diff) >= 2:
                _, pvalue = ttest_rel(joined["reference"], joined["baseline"])
                ci = float(t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / math.sqrt(len(diff)))
            else:
                pvalue, ci = np.nan, np.nan
            record = dict(zip(setting_columns, keys))
            record.update({"reference": reference, "baseline": method, "metric": metric, "n_pairs": len(diff), "mean_difference": float(diff.mean()) if len(diff) else np.nan, "ci95_halfwidth": ci, "p_value": pvalue})
            local_rows.append(record)
        adjusted = holm_bonferroni([r["p_value"] for r in local_rows])
        for row, value in zip(local_rows, adjusted):
            row["holm_bonferroni_p"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def write_artifacts(output_dir: str | Path, raw: RawData, config: ProtocolConfig, experiment_frames: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Save raw per-seed outputs, summaries, config, and a provenance manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    manifest = {"raw_data": raw.metadata, "protocol": asdict(config), "implementation_notes": {
        "pmf": "Not included: the cited PMF method assumes a mobility model with a risky location-embedding module and a personal adaptor. This tabular binary classifier has neither; calling a FedAvg variant PMF would be inaccurate.",
        "planar_laplace": "Implemented as a perturbation mechanism. It is not claimed to provide a formal global epsilon-geo-indistinguishability guarantee for a worldwide coordinate set.",
        "privacy_scope": "AIE/Re-ID evaluate only location-derived client coordination metadata against the declared nearest-candidate attacker; they do not measure model-update leakage.",
    }}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    paths["manifest"] = output / "manifest.json"
    for name, frame in experiment_frames.items():
        seed_path = output / f"{name}_seed_results.csv"
        frame.to_csv(seed_path, index=False)
        paths[f"{name}_seed"] = seed_path
        groups = [c for c in ("experiment", "partition", "n_clients", "k", "method", "mask", "mask_scale_km", "mask_level") if c in frame.columns]
        summary_path = output / f"{name}_summary.csv"
        summarize(frame, groups).to_csv(summary_path, index=False)
        paths[f"{name}_summary"] = summary_path
        if name == "main":
            paired_path = output / "main_paired_tests.csv"
            paired_tests(frame).to_csv(paired_path, index=False)
            paths["main_paired_tests"] = paired_path
    return paths


def run_smoke_test(raw: RawData, output_dir: str | Path) -> dict[str, Path]:
    """Fast execution check only; it is not a reportable experiment."""
    config = ProtocolConfig(rounds=2, local_epochs=1, seeds=(1,), client_counts=(8,), mask_scales_km=(0.5,), primary_mask_scale_km=0.5)
    main = main_comparison(raw, output_dir, config, partitions=("geographic",))
    return write_artifacts(output_dir, raw, config, {"main_smoke": main})
