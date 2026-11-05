"""Dataset loading, validation, target detection and leakage-safe feature selection."""

import os
import re

import numpy as np
import pandas as pd

TARGET_NAME_PATTERN = re.compile(r"(?:label|class|attack|target|category|type)", re.IGNORECASE)
LEAKAGE_NAME_PATTERN = re.compile(
    r"(?:label|attack|class|category|target|timestamp|^time$|^date|_id$|^id$|^index$|mac$|^device|"
    r"^src_ip$|^dst_ip$|ip_address|flow_id)",
    re.IGNORECASE,
)
BENIGN_PATTERN = re.compile(r"(?:benign|normal|legit)", re.IGNORECASE)
MAX_TARGET_CLASSES = 50
CHUNK_SIZE = 200_000


class DatasetError(ValueError):
    """Raised when an uploaded dataset cannot be used."""


def read_csv_sampled(source, max_rows: int, sample_fraction: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Read a CSV (path or file-like) in chunks, optionally sampling each chunk.

    Chunked reading keeps memory bounded for multi-GB files such as the full
    CIC-IIoT-2025 ``combined_dataset.csv``.
    """
    if max_rows <= 0:
        raise DatasetError("Maximum rows must be positive.")
    if not 0 < sample_fraction <= 1:
        raise DatasetError("Sample fraction must be in (0, 1].")
    if isinstance(source, (str, os.PathLike)):
        if not os.path.isfile(source):
            raise DatasetError(f"File not found: {source}")
        if not str(source).lower().endswith(".csv"):
            raise DatasetError("Only .csv files are supported.")

    rng = np.random.default_rng(seed)
    parts, total = [], 0
    try:
        reader = pd.read_csv(source, chunksize=CHUNK_SIZE, low_memory=False)
        for chunk in reader:
            if sample_fraction < 1:
                chunk = chunk.sample(frac=sample_fraction, random_state=int(rng.integers(0, 2**31 - 1)))
            parts.append(chunk)
            total += len(chunk)
            if total >= max_rows:
                break
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The CSV file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not parse the CSV file: {exc}") from exc

    if not parts:
        raise DatasetError("The CSV file contains no rows.")
    df = pd.concat(parts, ignore_index=True).head(max_rows)
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        raise DatasetError("The dataset needs at least one feature column and one label column.")
    if df.columns.duplicated().any():
        raise DatasetError("The dataset has duplicate column names.")
    return df


def read_uploaded(uploaded_file, max_rows: int, sample_fraction: float = 1.0) -> pd.DataFrame:
    if uploaded_file is None:
        raise DatasetError("No file uploaded.")
    if not uploaded_file.name.lower().endswith(".csv"):
        raise DatasetError("Please upload a .csv file.")
    uploaded_file.seek(0)
    return read_csv_sampled(uploaded_file, max_rows, sample_fraction)


def target_candidates(df: pd.DataFrame) -> list:
    """Rank columns that could be the class label. Nothing is assumed about the name."""
    scored = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique < 2 or nunique > MAX_TARGET_CLASSES:
            continue
        score = 0
        if TARGET_NAME_PATTERN.search(col):
            score += 2
        if not pd.api.types.is_numeric_dtype(df[col]):
            score += 2
        if df[col].astype(str).str.contains(BENIGN_PATTERN).any():
            score += 3
        scored.append((score, -abs(nunique - 8), col))
    scored.sort(reverse=True)
    return [c for _, _, c in scored]


def default_excluded_columns(df: pd.DataFrame, target: str) -> list:
    """Columns that are labels, identifiers or timestamps and would leak information."""
    excluded = [c for c in df.columns if c != target and LEAKAGE_NAME_PATTERN.search(c)]
    for c in df.columns:
        if c == target or c in excluded:
            continue
        if df[c].nunique(dropna=True) == len(df) and len(df) > 100 and not pd.api.types.is_float_dtype(df[c]):
            excluded.append(c)
    return excluded


def numeric_feature_columns(df: pd.DataFrame, target: str, excluded) -> list:
    excluded = set(excluded) | {target}
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            cols.append(c)
    return cols


def detect_benign_label(classes) -> str | None:
    for c in classes:
        if BENIGN_PATTERN.search(str(c)):
            return c
    return None


def prepare_xy(df: pd.DataFrame, target: str, features: list):
    """Return numeric X and string y, dropping rows without a label."""
    if target not in df.columns:
        raise DatasetError(f"Target column '{target}' not found.")
    if not features:
        raise DatasetError("No numeric feature columns are available after excluding leakage columns.")
    data = df[features + [target]].copy()
    data = data[data[target].notna()]
    X = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)
    y = data[target].astype(str).str.strip()
    if y.nunique() < 2:
        raise DatasetError("The target column must contain at least two classes.")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    inf = df.select_dtypes(include="number").apply(lambda s: np.isinf(s).sum())
    table = pd.DataFrame(
        {
            "Missing": missing,
            "Missing (%)": missing / max(len(df), 1) * 100,
            "Infinite": inf.reindex(df.columns).fillna(0).astype(int),
            "Dtype": df.dtypes.astype(str),
        }
    )
    return table.sort_values("Missing", ascending=False)


def make_synthetic_demo(n_rows: int = 6000, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC data for trying the app without the real dataset. Not CIC-IIoT-2025 data."""
    rng = np.random.default_rng(seed)
    classes = ["benign", "bruteforce", "ddos", "dos", "malware", "mitm", "recon", "web"]
    weights = np.array([0.55, 0.03, 0.09, 0.09, 0.05, 0.05, 0.11, 0.03])
    y = rng.choice(classes, size=n_rows, p=weights / weights.sum())
    names = [
        "network_interval-packets", "network_ip-flags_avg", "network_ips_all_count",
        "network_packets_all_count", "network_packets_dst_count", "network_time-delta_avg",
        "network_time-delta_min", "network_ttl_min", "network_window-size_std_deviation",
        "network_tcp-flags_avg", "network_window-size_min", "network_packet-size_avg",
        "network_payload-length_avg", "network_ports_all_count", "log_messages_count",
        "noise_feature_1", "noise_feature_2", "noise_feature_3",
    ]
    centers = rng.normal(0, 1.4, size=(len(classes), 15))
    idx = np.array([classes.index(v) for v in y])
    informative = centers[idx] + rng.normal(0, 1.0, size=(n_rows, 15))
    noise = rng.normal(0, 1, size=(n_rows, 3))
    df = pd.DataFrame(np.hstack([informative, noise]), columns=names)
    df.loc[rng.random(n_rows) < 0.01, "network_ttl_min"] = np.nan
    df["traffic_category"] = y
    df["timestamp"] = pd.date_range("2025-01-01", periods=n_rows, freq="s").astype(str)
    return df
