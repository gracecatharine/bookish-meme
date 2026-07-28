
"""
notes: this is the code that works with baronwind format
will need the reference dataset to be renamed to reference_copy and in CSV format
headers are hardcoded in here to match the reference dataset
outputs are stored in a folder called pipeline_outputs which gets generated when run
masttool.env
"""

#libraries and packages etc
import argparse
import ast
import glob
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

#general configurations
output_dir_default = "pipeline_outputs"
reference_csv_default = "reference_copy.csv"

#assign values
wind_reference_col = "Speed_100m [m/s]"
temp_reference_col = "Temperature_2m [degrees C]"
reference_time_col = "Date/time [UTC]"

mast_id_regexes = [
    r"target[_-]?(\d{3,5})",
    r"(\d{3,5})[_-]?target",
    r"mast[_-]?(\d{3,5})",
    r"(\d{3,5})",
]

#running statements
excluded_output_tokens = (
    "corrected",
    "diagnostic",
    "window",
    "elevation",
    "density",
    "merged",
    "summary",
    "ratio",
    "heatmap",
    "monthly",
    "mad",
    "scaling",
    "ti",
    "pipeline_outputs",
)

#set defaults
default_heights = [50, 80, 110, 140]
primary_speed_height = 110

wind_ratio_cap_min = 0.7
wind_ratio_cap_max = 1.3

#target window is inferred from the mast input files
default_target_start = None
default_target_end = None

#general helpers
#defining data containers
@dataclass
class PipelineConfig:
    output_dir: str = output_dir_default
    reference_csv: str = reference_csv_default
    overlap_start: pd.Timestamp = None
    overlap_end: pd.Timestamp = None
    target_start: pd.Timestamp = default_target_start
    target_end: pd.Timestamp = default_target_end
    heights: list = None
    primary_speed_height: int = primary_speed_height
    wind_ratio_cap_min: float = wind_ratio_cap_min
    wind_ratio_cap_max: float = wind_ratio_cap_max
    #when False skips writing some pathways csvs
    write_diagnostics: bool = False

#check overlaps
    def __post_init__(self):
        if self.heights is None:
            self.heights = default_heights.copy()
        if self.overlap_start is not None:
            self.overlap_start = pd.to_datetime(self.overlap_start)
        if self.overlap_end is not None:
            self.overlap_end = pd.to_datetime(self.overlap_end)
        if self.target_start is not None:
            self.target_start = pd.to_datetime(self.target_start)
        if self.target_end is not None:
            self.target_end = pd.to_datetime(self.target_end)

#return path
    def path(self, filename):
        return os.path.join(self.output_dir, filename)

@dataclass
class MastRecord:
    mast_id: str
    raw_path: str
    config: PipelineConfig

    @property
    def wind_corrected_path(self):
        return self.config.path(f"wind_corrected_{self.mast_id}.csv")

    @property
    def wind_diagnostic_path(self):
        return self.config.path(f"wind_scaling_diagnostic_{self.mast_id}.csv")

    @property
    def best_window_path(self):
        return self.config.path(f"best_window_corrected_{self.mast_id}.csv")

    @property
    def temp_corrected_path(self):
        return self.config.path(f"temp_corrected_{self.mast_id}.csv")

    @property
    def elevation_path(self):
        return self.config.path(f"elevation_{self.mast_id}.csv")

    @property
    def ti_corrected_path(self):
        return self.config.path(f"ti_corrected_{self.mast_id}.csv")

    @property
    def merged_path(self):
        return self.config.path(f"merged_timeseries_{self.mast_id}.csv")

def ensure_output_dir(config):
    os.makedirs(config.output_dir, exist_ok=True)

def normalize_month(m):
    if pd.isna(m):
        return None
    try:
        m_int = int(m)
        if 1 <= m_int <= 12:
            return m_int
    except (ValueError, TypeError):
        pass
    try:
        return pd.to_datetime(str(m), format="%B").month
    except Exception:
        pass
    try:
        return pd.to_datetime(str(m)).month
    except Exception:
        pass
    return None

def infer_target_window_from_masts(masts):
    bounds = []
    for mast in masts:
        df = load_mast_input(mast.raw_path)
        df = build_timestamp(df)
        if "_timestamp" not in df.columns:
            continue

        timestamps = pd.to_datetime(df["_timestamp"]).dropna()
        if timestamps.empty:
            continue
        bounds.append((timestamps.min(), timestamps.max()))

    start = max(bound[0] for bound in bounds)
    end = min(bound[1] for bound in bounds)

    if start >= end:
        raise ValueError("The mast files do not overlap in time")

    start = start.to_period("M").to_timestamp()
    end = end.to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)
    end = end.replace(hour=23, minute=59, second=59, microsecond=0)
    return start, end

#loop thru to find mast id numbers
def infer_mast_id(path):
    stem = Path(path).stem
    for pattern in mast_id_regexes:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    raise ValueError(f"{path}")

#id a mast file
def is_probable_raw_mast_file(path):
    basename = os.path.basename(path).lower()
    if not basename.endswith(".csv"):
        return False
    # avoid false matches
    for token in excluded_output_tokens:
        pattern = rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
        if re.search(pattern, basename):
            return False
    if basename == reference_csv_default.lower():
        return False
    return True

def discover_masts(config, targets=None, target_glob=None):
    paths = []
    if targets:
        paths.extend(targets)
    if target_glob:
        paths.extend(sorted(glob.glob(target_glob)))
    if not paths:
        for pattern in (
            "*_target_*.csv",
            "target_*.csv",
            "*_Mast_*.csv",   
            "*_mast_*.csv",
            "*Mast*.csv",    
            "*mast*.csv",
        ):
            paths.extend(sorted(glob.glob(pattern)))

    masts, seen = [], set()
    for path in paths:
        key = os.path.abspath(path)
        if key in seen:
            continue
        seen.add(key)
        if not is_probable_raw_mast_file(path):   
            continue
        masts.append(
            MastRecord(mast_id=infer_mast_id(path), raw_path=path, config=config)
        )
    # failing glob go back and look
    if not masts:
        for candidate in sorted(glob.glob("*.csv")):
            key = os.path.abspath(candidate)
            if key in seen:
                continue
            seen.add(key)
            if not is_probable_raw_mast_file(candidate):
                continue
            try:
                masts.append(MastRecord(mast_id=infer_mast_id(candidate), raw_path=candidate, config=config))
            except Exception:
                continue

    if not masts:
        raise FileNotFoundError(
            "No mast input CSVs found (looked for *_Mast_*.csv / *_target_*.csv etc.)."
        )
    return masts

def compute_global_overlap(config, masts):
    df_reference = load_reference(config.reference_csv)

    overlap_starts = []
    overlap_ends = []

    for mast in masts:
        df = load_mast_input(mast.raw_path)
        df = build_timestamp(df)
        df.set_index("_timestamp", inplace=True)
        df.sort_index(inplace=True)

        start, end = compute_overlap(df, df_reference)
        overlap_starts.append(start)
        overlap_ends.append(end)

    global_start = max(overlap_starts)
    global_end = min(overlap_ends)

    # enforce ≥ 12 months
    n_months = (global_end.year - global_start.year) * 12 + (global_end.month - global_start.month) + 1
    if n_months < 12:
        raise ValueError("Overlap is less than 12 months")
    return global_start, global_end

def build_timestamp(df):
    df = df.copy()
    if "_timestamp" in df.columns:
        df["_timestamp"] = pd.to_datetime(df["_timestamp"])
        return df
    if {"Year", "Month", "Day", "Hour", "Minute"}.issubset(df.columns):
        df["_timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
        return df
    if {"Year", "Month", "Day", "Hour"}.issubset(df.columns):
        df["_timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour"]])
        return df
    if {"Year", "Month", "Day"}.issubset(df.columns):
        df["_timestamp"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        return df
    if reference_time_col in df.columns:
        df["_timestamp"] = pd.to_datetime(df[reference_time_col])
        return df
    raise ValueError()

def load_timeseries_with_filter(path, start_date, end_date):
    df = pd.read_csv(path)
    df = build_timestamp(df)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return df[(df["_timestamp"] >= start_date) & (df["_timestamp"] <= end_date)].copy()

def monthly_climatology(df, column):
    return df.groupby(df.index.month)[column].mean().reindex(range(1, 13)).to_numpy()

def load_reference(reference_csv):
    df = pd.read_csv(reference_csv)
    required = [reference_time_col, wind_reference_col, temp_reference_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{missing}")
    df[reference_time_col] = pd.to_datetime(df[reference_time_col])
    df.set_index(reference_time_col, inplace=True)
    df.sort_index(inplace=True)
    return df

def discover_speed_columns(df):
    speed_columns = []
    for column_name in df.columns:
        match = re.fullmatch(r"Speed (\d+)m syn \[m/s\]", column_name)
        if match:
            speed_columns.append((int(match.group(1)), column_name))
    speed_columns.sort(key=lambda item: item[0])
    return speed_columns

def discover_temperature_column(df, height):
    expected = f"Temperature {height}m syn [°C]"
    if expected in df.columns:
        return expected

    temp_cols = [c for c in df.columns if "temperature" in c.lower()]
    for c in temp_cols:
        c_lower = c.lower()
        if f"{height}m" in c_lower or f"{height} m" in c_lower:
            return c
    return None

def parse_ratio_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value)

def read_input_header_block(path):
#returns metadata and header lines from mast file csv
    meta_lines, header_line = [], None
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.rstrip("\n")
            cells = [c.strip() for c in stripped.split(",")[:5]]
            if cells == ["Year", "Month", "Day", "Hour", "Minute"]:
                header_line = stripped        # the column header row
                break
            meta_lines.append(stripped)        # MM / lat-lon / heights / etc.
    if header_line is None:
        raise ValueError(
            f"No 'Year,Month,Day,Hour,Minute' header row found in {path}"
        )
    return meta_lines, header_line

def load_mast_input(path):
#read inputs
    try:
        df = pd.read_csv(path)
    except Exception:
        meta_lines, header_line = read_input_header_block(path)
        skip = len(meta_lines)
        df = pd.read_csv(path, skiprows=skip)
        return df

#incase reading not working
    if not ({"Year", "Month", "Day"}.issubset(df.columns) or reference_time_col in df.columns or "_timestamp" in df.columns):
        try:
            meta_lines, header_line = read_input_header_block(path)
            skip = len(meta_lines)
            df = pd.read_csv(path, skiprows=skip)
        except Exception:
            pass
    return df

#stage 1 climatology
def stage1_climatology(config):
    ensure_output_dir(config)
    df_reference = load_reference(config.reference_csv)
    df_overlap_ref = df_reference.loc[config.overlap_start:config.overlap_end]

    ref_monthly_mean = (
        df_reference.groupby(df_reference.index.month)[wind_reference_col]
        .mean()
        .reset_index()
    )
    ref_monthly_mean.columns = ["Month", wind_reference_col]
    ref_monthly_mean.to_csv(config.path("ref_monthly_mean.csv"), index=False)

    overlap_monthly_mean = (
        df_overlap_ref.groupby(df_overlap_ref.index.month)[wind_reference_col   ]
        .mean()
        .reset_index()
    )
    overlap_monthly_mean.columns = ["Month", wind_reference_col]
    overlap_monthly_mean.to_csv(config.path("overlap_ref_monthly_mean.csv"), index=False)

    ref_monthly_temp = (
        df_reference.groupby(df_reference.index.month)[temp_reference_col]
        .mean()
        .reset_index()
    )
    ref_monthly_temp.columns = ["Month", temp_reference_col]
    ref_monthly_temp.to_csv(config.path("ref_monthly_temp.csv"), index=False)

    overlap_monthly_temp = (
        df_overlap_ref.groupby(df_overlap_ref.index.month)[temp_reference_col]
        .mean()
        .reset_index()
    )
    overlap_monthly_temp.columns = ["Month", temp_reference_col]
    overlap_monthly_temp.to_csv(config.path("overlap_ref_monthly_temp.csv"), index=False)

    print("Wrote reference climatology files to", config.output_dir)

#stage 2 scaling
def load_wind_ratio_data(config):
    ratio_path = config.path("ref_monthly_mean.csv")
    overlap_path = config.path("overlap_ref_monthly_mean.csv")
    long_term = pd.read_csv(ratio_path)
    overlap = pd.read_csv(overlap_path)

    ratio_df = pd.DataFrame({
        "Month": long_term["Month"],
        "Reference_Mean": long_term[wind_reference_col],
        "Overlap_Mean": overlap[wind_reference_col],
    })
    ratio_df["Month"] = ratio_df["Month"].apply(normalize_month)
    if ratio_df["Month"].isna().any():
        raise ValueError()

    ratio_df["Original_Ratio"] = ratio_df["Reference_Mean"] / ratio_df["Overlap_Mean"]
    ratio_df["Applied_Ratio"] = ratio_df["Original_Ratio"].clip(
        lower=config.wind_ratio_cap_min,
        upper=config.wind_ratio_cap_max,
    )
    ratio_df["Capped"] = ratio_df["Original_Ratio"] != ratio_df["Applied_Ratio"]
    return ratio_df

def apply_wind_scaling(df_target, ratio_df, primary_height=primary_speed_height, mean_override=None):
    ratio_map = dict(zip(ratio_df["Month"].astype(int), ratio_df["Applied_Ratio"].astype(float)))

    months = df_target["Month"].apply(normalize_month)
    if months.isna().any():
        raise ValueError()

    monthly_scale = months.map(ratio_map)
    if monthly_scale.isna().any():
        missing_months = sorted(months[monthly_scale.isna()].unique())
        raise ValueError(f"{missing_months}")

    df_target = df_target.copy()
    df_target["wind_applied_ratio"] = monthly_scale

    speed_columns = discover_speed_columns(df_target)
    if not speed_columns:
        raise ValueError()

    primary_speed_col = None
    for height, speed_col in speed_columns:
        if height == primary_height:
            primary_speed_col = speed_col
            break
    if primary_speed_col is None:
        primary_speed_col = speed_columns[0][1]

    diagnostic = None
    for height, speed_col in speed_columns:
        original_speed = df_target[speed_col].astype(float)
        scaled_speed = original_speed * monthly_scale
        original_mean = original_speed.mean()
        desired_mean = original_mean if mean_override is None else float(mean_override)
        scaled_mean = scaled_speed.mean()
        global_factor = desired_mean / scaled_mean if scaled_mean else 1.0
        corrected_speed = scaled_speed * global_factor

        df_target[f"Scaled_Speed_{height}"] = scaled_speed
        df_target[f"Corrected_Speed_{height}"] = corrected_speed

        if speed_col == primary_speed_col:
            df_target["Scaled_Speed"] = scaled_speed
            df_target["Corrected_Speed"] = corrected_speed
            diagnostic = pd.DataFrame({
                "Metric": [
                    "Original Mean",
                    "Scaled before restoration",
                    "Desired Mean",
                    "Global Restoration Factor",
                    "Number of capped months",
                ],
                "Value": [
                    original_mean,
                    scaled_mean,
                    desired_mean,
                    global_factor,
                    int(ratio_df["Capped"].sum()),
                ],
            })
    return df_target, diagnostic

def run_stage2(config, targets=None, target_glob=None):
    ensure_output_dir(config)
    masts = discover_masts(config, targets=targets, target_glob=target_glob)
    ratio_df = load_wind_ratio_data(config)
    if config.write_diagnostics:
        ratio_df.to_csv(config.path("wind_scaling_ratios.csv"), index=False)

    for mast in masts:
        print(f"wind correction for mast {mast.mast_id}: {mast.raw_path}")
        df_target = load_mast_input(mast.raw_path)
        df_target = build_timestamp(df_target)
        df_target, diagnostic = apply_wind_scaling(
            df_target,
            ratio_df,
            primary_height=config.primary_speed_height,
        )
        df_target.to_csv(mast.wind_corrected_path, index=False)
        print(f"  wrote {mast.wind_corrected_path}")
        if config.write_diagnostics:
            diagnostic.to_csv(mast.wind_diagnostic_path, index=False)
            print(f"  wrote {mast.wind_diagnostic_path}")

#stage 3 best window
def compute_overlap(df1, df2):
    start = max(df1.index.min(), df2.index.min())
    end = min(df1.index.max(), df2.index.max())
    
    if start >= end:
        raise ValueError()

    if start.day != 1 or start.hour != 0 or start.minute != 0:
        start = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + pd.DateOffset(months=1)
    else:
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)

    expected_month_end = end + pd.offsets.MonthEnd(0)
    if end.date() != expected_month_end.date():
        end = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(minutes=1)
    else:
        end = expected_month_end.replace(hour=23, minute=59, second=59, microsecond=0)
    if start >= end:
        raise ValueError()
    return start, end

def enumerate_windows(overlap_start, overlap_end, window_months=12):
    anchors = []
    collect = overlap_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while collect <= overlap_end:
        anchors.append(collect)
        collect += pd.DateOffset(months=1)
    windows = []

    for i in range(len(anchors) - window_months + 1):
        w_start = anchors[i]
        w_end_monthstart = anchors[i + window_months - 1]
        w_end = w_end_monthstart + pd.offsets.MonthEnd(0)
        w_end = w_end.replace(hour=23, minute=59, second=59, microsecond=0)
        windows.append((w_start, w_end))
    return windows

def monthly_mean_window(df, column, win_start, win_end):
    subset = df.loc[win_start:win_end, column].dropna()
    return subset.groupby(subset.index.month).mean().reindex(range(1, 13))

def compute_ratio(longterm_series, window_series, cap_min, cap_max):
    orig = (longterm_series / window_series).values
    clipped = np.clip(orig, cap_min, cap_max)
    capped = orig != clipped
    return orig, clipped, capped

def mad_score(applied_ratios):
    return np.nanmean(np.abs(applied_ratios - 1.0))

def run_stage3(config, targets=None, target_glob=None, speed_col=None):
    ensure_output_dir(config)
    speed_col = speed_col or f"Speed {config.primary_speed_height}m syn [m/s]"
    # find raw mast files 
    masts = discover_masts(config, targets=targets, target_glob=target_glob)
    # load the long-term reference 
    df_reference = load_reference(config.reference_csv)
    # compute the long-term monthly wind climatology 
    longterm_mean = df_reference.groupby(df_reference.index.month)[wind_reference_col].mean().reindex(range(1, 13))

    overlap_start = config.overlap_start
    overlap_end = config.overlap_end
    windows = enumerate_windows(overlap_start, overlap_end)

    if not windows:
        raise ValueError()

    print(f"Overlap start: {overlap_start}")
    print(f"Overlap end: {overlap_end}")
    print(f"Total windows: {len(windows)}")

    records = []
   
    #storing mads
    for win_start, win_end in windows:
        label = f"{win_start.strftime('%b%Y')}-{win_end.strftime('%b%Y')}"
        target_ratios = {}
        target_capped = []

        reference_window_mean = monthly_mean_window(df_reference, wind_reference_col, win_start, win_end)
        orig_ratios, applied_ratios, capped_flags = compute_ratio(
            longterm_mean,
            reference_window_mean,
            config.wind_ratio_cap_min,
            config.wind_ratio_cap_max
        )

        combined_mad = mad_score(applied_ratios)

        n_capped_months = int(np.sum(capped_flags & ~np.isnan(orig_ratios)))
        record = {
            "window": label,
            "Start": win_start,
            "End": win_end,
            "Combined_MAD": round(float(combined_mad), 6),
            "N_capped_months": n_capped_months,
            "Applied_ratios": applied_ratios.tolist(),
        }

        for mast in masts:
            record[f"MAD_{mast.mast_id}"] = round(float(combined_mad), 6)
        record.update(target_ratios)
        records.append(record)

    results_df = pd.DataFrame(records)
    results_df.sort_values("Combined_MAD", inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    results_df.insert(0, "Rank", results_df.index + 1)
    results_df.to_csv(config.path("window_diagnostics.csv"), index=False)
    print("Saved", config.path("window_diagnostics.csv"))

    #best selection
    best = results_df.iloc[0]
    best_start = pd.Timestamp(best["Start"])
    best_end = pd.Timestamp(best["End"])
    print("\nSummary")
    print(f"Best window : {best['window']}")
    print(f"Combined MAD: {best['Combined_MAD']:.6f}")
    print(f"Capped months in best window: {best['N_capped_months']}")

    for mast in masts:
        best_applied = parse_ratio_list(best["Applied_ratios"])
        best_ratio_map = {
            month: ratio
            for month, ratio in zip(range(1, 13), best_applied)
            if not pd.isna(ratio)
        }

        # full target data before trimming
        df_full = load_mast_input(mast.raw_path)
        df_full = build_timestamp(df_full)

        full_speed_columns = discover_speed_columns(df_full)
        if not full_speed_columns:
            raise ValueError(f"{mast.mast_id}: no Speed {{height}}m syn [m/s] columns found")
        full_target_mean = {
            height: df_full[col].astype(float).mean()
            for height, col in full_speed_columns
        }

        df_target = df_full[
            (df_full["_timestamp"] >= best_start) &
            (df_full["_timestamp"] <= best_end)
        ].copy()

        if df_target.empty:
            raise ValueError(f"{mast.mast_id}: no data in best window")

        months_col = df_target["Month"].apply(normalize_month).astype(int)
        monthly_scale = months_col.map(best_ratio_map)

        if monthly_scale.isna().any():
            missing = sorted(months_col[monthly_scale.isna()].unique())
            raise ValueError(f"{mast.mast_id}: missing best-window ratios for months {missing}")

        df_target["best_window_applied_ratio"] = monthly_scale
        speed_columns = discover_speed_columns(df_target)
        if not speed_columns:
            raise ValueError(f"{mast.mast_id}: no Speed {{height}}m syn [m/s] columns found")

        for height, height_speed_col in speed_columns:
            original_speed = df_target[height_speed_col].astype(float)
            scaled_speed = original_speed * monthly_scale
            # retore to overall mean
            desired_mean = full_target_mean[height]
            scaled_mean = scaled_speed.mean()
            global_factor = desired_mean / scaled_mean if scaled_mean else 1.0
            corrected_speed = scaled_speed * global_factor
            df_target[f"Scaled_Speed_{height}"] = scaled_speed
            df_target[f"Corrected_Speed_{height}"] = corrected_speed
            df_target[f"speed_final_scale_factor_{height}"] = global_factor
            df_target[f"speed_full_target_mean_{height}"] = desired_mean
            df_target[f"speed_scaled_window_mean_{height}"] = scaled_mean

            if height == config.primary_speed_height:
                df_target["Scaled_Speed"] = scaled_speed
                df_target["Corrected_Speed"] = corrected_speed
                df_target["best_window_scaled_speed"] = scaled_speed
                df_target["best_window_corrected_speed"] = corrected_speed

        df_target.to_csv(mast.best_window_path, index=False)
        print(f"Saved {mast.best_window_path}")

#stage 4 temperature
def load_temperature_ratio_data(config, best_start, best_end):
    df_reference = load_reference(config.reference_csv)
    long_term = (
        df_reference
        .groupby(df_reference.index.month)[temp_reference_col]
        .mean()
        .reindex(range(1, 13))
    )
    window = df_reference.loc[best_start:best_end]

    window_mean = (
        window
        .groupby(window.index.month)[temp_reference_col]
        .mean()
        .reindex(range(1, 13))
    )
    ratio_df = pd.DataFrame({
        "Month": range(1, 13),
        "Reference_Temp_K": long_term + 273.15,
        "Window_Temp_K": window_mean + 273.15,
    })
    ratio_df["Ratio"] = (
        ratio_df["Reference_Temp_K"] / ratio_df["Window_Temp_K"]
    )
    return ratio_df

def read_target_for_temperature(path, heights):
    df_target = load_mast_input(path)
    for h in heights:
        source_col = discover_temperature_column(df_target, h)
        if source_col is None:
            temp_cols = [c for c in df_target.columns if "temperature" in c.lower()]
            raise KeyError(
                f"missing {h}m in {path}. "
                f"{temp_cols}"
            )
        df_target[f"Temperature_k_{h}"] = df_target[source_col].astype(float) + 273.15
    return df_target

def apply_temperature_scaling(df_target, ratio_df, heights, full_temp_mean_k_by_height=None):
#temp scaling, then back down to mean
    ratio_map = dict(zip(ratio_df["Month"].astype(int), ratio_df["Ratio"].astype(float)))
    months = df_target["Month"].apply(normalize_month)

    monthly_scale = months.map(ratio_map)

    df_target = df_target.copy()
    df_target["temperature_applied_ratio"] = monthly_scale

    for h in heights:
        source_k_col = f"Temperature_k_{h}"
        original_temp_k = df_target[source_k_col].astype(float)
        scaled_temp_k = original_temp_k * monthly_scale

        if full_temp_mean_k_by_height is not None and h in full_temp_mean_k_by_height:
            desired_mean_k = float(full_temp_mean_k_by_height[h])
        else:
            desired_mean_k = float(original_temp_k.mean())

        scaled_mean_k = float(scaled_temp_k.mean())
        global_factor = desired_mean_k / scaled_mean_k if scaled_mean_k and not pd.isna(scaled_mean_k) else 1.0
        corrected_temp_k = scaled_temp_k * global_factor

        #scaled temps
        df_target[f"scaled_temp_{h}K"] = scaled_temp_k
        df_target[f"scaled_temp_{h}C"] = scaled_temp_k - 273.15

        #restored to mean
        df_target[f"corrected_temp_{h}K"] = corrected_temp_k
        df_target[f"corrected_temp_{h}C"] = corrected_temp_k - 273.15
        df_target[f"temperature_final_scale_factor_{h}"] = global_factor
        df_target[f"temperature_full_target_mean_K_{h}"] = desired_mean_k
        df_target[f"temperature_scaled_window_mean_K_{h}"] = scaled_mean_k

    return df_target

def run_stage4(config, targets=None, target_glob=None):
    ensure_output_dir(config)
    masts = discover_masts(config, targets=targets, target_glob=target_glob)

    #read stage3 window
    window_df = pd.read_csv(config.path("window_diagnostics.csv"))
    best = window_df.sort_values("Rank").iloc[0]
    best_start = pd.Timestamp(best["Start"])
    best_end = pd.Timestamp(best["End"])

    ratio_df = load_temperature_ratio_data(
        config,
        best_start,
        best_end,
    )

    for mast in masts:
        print(f"Stage 4 temperature correction for mast {mast.mast_id}: {mast.raw_path}")

        # store mean temps from full before trim
        df_full = load_mast_input(mast.raw_path)
        df_full = build_timestamp(df_full)
        full_temp_mean_k_by_height = {}
        for h in config.heights:
            source_col = discover_temperature_column(df_full, h)
            full_temp_mean_k_by_height[h] = float(df_full[source_col].astype(float).mean() + 273.15)

        #stage 3 best window applied
        df_target = pd.read_csv(mast.best_window_path)
        df_target = build_timestamp(df_target)

        for h in config.heights:
            source_col = discover_temperature_column(df_target, h)
            df_target[f"Temperature_k_{h}"] = df_target[source_col].astype(float) + 273.15

        #monthly scale and restore
        df_target = apply_temperature_scaling(
            df_target,
            ratio_df,
            config.heights,
            full_temp_mean_k_by_height=full_temp_mean_k_by_height,
        )
        df_target = build_timestamp(df_target)
        df_target.to_csv(mast.temp_corrected_path, index=False)
        print(f"  wrote {mast.temp_corrected_path}")

#stage 5 elevation
def elevation(rho, T_k, P0=101325, R=287.05, g=-9.80665):
    
    if pd.isna(rho) or pd.isna(T_k) or T_k <= 0:
        return np.nan

    K = g / (R * T_k)
    a = 0.000025 * K
    b = -1.0397 * K
    c = np.log(rho * (R * T_k) / P0)
    d = b ** 2 - 4 * a * c

    if d < 0:
        return np.nan

    z1 = (-b + np.sqrt(d)) / (2 * a)
    z2 = (-b - np.sqrt(d)) / (2 * a)
    positives = [z for z in (z1, z2) if z >= 0]

    if not positives:
        return np.nan
    return min(positives)

def compute_elevation(df, heights):
    df = df.copy()
    for h in heights:
        temp_col = f"Temperature_k_{h}"
        density_col = f"Density {h}m [kg/m³]"
        elevation_col = f"elevation_{h}m"
        mast_considered_col = f"mast_considered_elevation_{h}m"
        df[elevation_col] = df.apply(lambda row: elevation(row[density_col], row[temp_col]), axis=1)
        df[mast_considered_col] = df[elevation_col] - h
    return df

def compute_mean_elevation(all_dfs, heights):
    records = []
    for label, df in all_dfs:
        row = {"mast_id": label}
        for h in heights:
            elevation_col = f"elevation_{h}m"
            row[f"avg_elevation_{h}m"] = df[elevation_col].mean(skipna=True) if elevation_col in df.columns else np.nan
        records.append(row)
    return pd.DataFrame(records)

def recompute_density(df, heights, avg_elevation, P0=101325, R=287.05, g=-9.80665):
#elevation is back-calculated from the pre-scaled temperature
#recomputes density
    df = df.copy()

    for h in heights:
        temp_col = f"corrected_temp_{h}K"
        new_density_col = f"recomputed_density_{h}m"

        z = avg_elevation[h]
        if pd.isna(z):
            df[new_density_col] = np.nan
            continue

        K = g / (R * df[temp_col].astype(float))
        a = 0.000025 * K
        b = -1.0397 * K
        exponent = -(a * (z ** 2) + b * z)
        density = (P0 / (R * df[temp_col].astype(float))) * np.exp(exponent)
        density[df[temp_col].isna() | (df[temp_col] <= 0)] = np.nan
        df[new_density_col] = density
    return df

def run_stage5(config, targets=None, target_glob=None):
    ensure_output_dir(config)
    masts = discover_masts(config, targets=targets, target_glob=target_glob)
    all_dfs = []

    for mast in masts:
        #read the full temp_corrected file and slice to the target window in-memory
        input_path = mast.temp_corrected_path
        print(f"Stage 5 elevation for mast {mast.mast_id}: {input_path}")
        best_df = pd.read_csv(mast.best_window_path)
        best_df = build_timestamp(best_df)
        start_date = best_df["_timestamp"].min()
        end_date = best_df["_timestamp"].max()
        df = load_timeseries_with_filter(
            input_path,
            start_date,
            end_date
        )
        print(f"  rows after date filter: {len(df)}")
        df = compute_elevation(df, config.heights)

#average elevation
        avg_elev = {
            h: df[f"elevation_{h}m"].mean(skipna=True)
            for h in config.heights
        }
        for h in config.heights:
            df[f"avg_elevation_{h}m"] = avg_elev[h]

        df = recompute_density(df, config.heights, avg_elev)

        df.to_csv(mast.elevation_path, index=False)
        print(f"  wrote {mast.elevation_path}")
        print(f"  averaged elevation ({mast.mast_id}): {avg_elev}")
        all_dfs.append((mast.mast_id, df))

    summary_df = compute_mean_elevation(all_dfs, config.heights)
    summary_path = config.path("mean_elevation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_df)
    print(f"wrote {summary_path}")

#stage 6 turblence
def find_corrected_speed_column(df, height=None):
    if height is not None:
        h = str(height)
        preferred = [
            f"Corrected_Speed_{h}",
            f"Scaled_Speed_{h}",
            "Corrected_Speed",
            "Scaled_Speed",
            "best_window_corrected_speed",
            "best_window_scaled_speed",
        ]
        for col in preferred:
            if col in df.columns:
                return col
    matches = [
        c for c in df.columns
        if "speed" in c.lower() and ("corrected" in c.lower() or "scaled" in c.lower())
    ]
    corrected = [c for c in matches if "corrected" in c.lower()]
    if corrected:
        return corrected[0]
    if matches:
        return matches[0]
    return None

def compute_ti_columns(df, heights):
    df = df.copy()
    for h in heights:
        speed_col = f"Speed {h}m syn [m/s]"
        ti_pct_col = f"Speed {h}m syn TI [%]"
        std_dev_col = f"stddev_{h}m"
        new_ti_col = f"new_TI_{h}m"

        df[std_dev_col] = (df[ti_pct_col].astype(float) / 100.0) * df[speed_col].astype(float)
        corrected_speed_col = find_corrected_speed_column(df, h)

        df[new_ti_col] = (df[std_dev_col] / df[corrected_speed_col].astype(float)) * 100.0
    return df

def run_stage6(config, targets=None, target_glob=None):
    ensure_output_dir(config)
    masts = discover_masts(config, targets=targets, target_glob=target_glob)

    for mast in masts:
#use best window!!
        input_path = mast.best_window_path

        print(f"Stage 6 TI correction for mast {mast.mast_id}: {input_path}")

        best_df = pd.read_csv(mast.best_window_path)
        best_df = build_timestamp(best_df)
        start_date = best_df["_timestamp"].min()
        end_date = best_df["_timestamp"].max()
        df = load_timeseries_with_filter(
            input_path,
            start_date,
            end_date
        )
        df = compute_ti_columns(df, config.heights)
        df.to_csv(mast.ti_corrected_path, index=False)
        print(f"  wrote {mast.ti_corrected_path}")

#stage 7 merge
def build_merged_frame(mast, config):
    wind_path = mast.ti_corrected_path
    temp_path = mast.temp_corrected_path
    elevation_path = mast.elevation_path

    df_wind = build_timestamp(pd.read_csv(wind_path))
    df_temp = build_timestamp(pd.read_csv(temp_path))
    df_elevation = build_timestamp(pd.read_csv(elevation_path))
    #use best window!!
    best_window_path = config.path(f"best_window_corrected_{mast.mast_id}.csv")

    df_base = build_timestamp(pd.read_csv(best_window_path))
    print(
        f"{mast.mast_id} timeline:",
        df_base["_timestamp"].min(),
        "->",
        df_base["_timestamp"].max()
     )

    df_merged = df_base.copy()
    for right_df, label in (
        (df_wind, "wind"),
        (df_temp, "temperature"),
        (df_elevation, "elevation_density"),
    ):
        cols_to_use = ["_timestamp"] + [
            c for c in right_df.columns
            if c not in df_merged.columns and c != "_timestamp"
        ]
        df_merged = pd.merge(
            df_merged,
            right_df[cols_to_use],
            on="_timestamp",
            how="left"
        )
        print(f"  merged {label} columns")
     #force date fields to match best window
    df_merged["Year"] = df_merged["_timestamp"].dt.year
    df_merged["Month"] = df_merged["_timestamp"].dt.month
    df_merged["Day"] = df_merged["_timestamp"].dt.day
    df_merged["Hour"] = df_merged["_timestamp"].dt.hour
    df_merged["Minute"] = df_merged["_timestamp"].dt.minute
    return df_merged

def run_stage7(config, targets=None, target_glob=None):
    ensure_output_dir(config)
    masts = discover_masts(config, targets=targets, target_glob=target_glob)
    for mast in masts:
        print(f"Stage 7 merge for mast {mast.mast_id}")
        df_merged = build_merged_frame(mast, config)
        write_final_timeseries(df_merged, mast.mast_id, config, mast.raw_path)
        if config.write_diagnostics:
            df_merged.to_csv(mast.merged_path, index=False)
            print(
                f"  wrote {mast.merged_path} "
                f"with {len(df_merged.columns)} columns and {len(df_merged)} rows"
            )

#stage 7 helpers - reconstructed-format final output
def reconstructed_template(mast_id):
    patterns = [
        f"*{mast_id}*econstructed*.csv",   
        f"*{mast_id}*Recon*.csv",
        f"*{mast_id}*.csv",
    ]
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if metadata_header(candidate):
                return candidate
    return None

def metadata_header(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            first = f.readline().strip()
        return first.split(",")[0].strip() == "MM"
    except Exception:
        return False

def read_metadata_header(path):
    meta_lines = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            cells = [c.strip() for c in line.rstrip("\n").split(",")[:5]]
            if cells == ["Year", "Month", "Day", "Hour", "Minute"]:
                break
            meta_lines.append(line.rstrip("\n"))
    return meta_lines

def default_metadata(mast_id, heights, n_cols):
    def pad(values):
        row = list(values) + [""] * (n_cols - len(values))
        return ",".join(str(v) for v in row)
    return [
        pad(["MM", "2.1"]),
        pad(["", "", ""]),                       # lat, lon, tz unknown
        pad(heights),                            # measurement heights
        pad([2] * len(heights)),
        pad([1.4] * len(heights)),
        pad(["3.01"]),
        pad([f"{mast_id}_Recon", f"final_timeseries_{mast_id}"]),
    ]

def find_corrected_temperature_column(df, height):
    candidates = [
        f"corrected_temp_{height}C",
        f"corrected_temp_{height}K",
        f"scaled_temp_{height}C",
        f"scaled_temp_{height}K",
        f"Temperature_k_{height}",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None

def write_final_timeseries(df_merged, mast_id, config, raw_input_path):
    df = df_merged.copy()
    #header
    meta_lines, header_line = read_input_header_block(raw_input_path)
    out_columns = [c.strip() for c in header_line.split(",")]

    #timestamp 
    if "_timestamp" in df.columns:
        ts = pd.to_datetime(df["_timestamp"])
        df["Year"], df["Month"], df["Day"] = ts.dt.year, ts.dt.month, ts.dt.day
        df["Hour"], df["Minute"] = ts.dt.hour, ts.dt.minute

    #replace w calculated values
    def resolve(col):
        # scaled/corrected wind speed
        m = re.fullmatch(r"Speed (\d+)m syn \[m/s\]", col)
        if m:
            src = find_corrected_speed_column(df, height=int(m.group(1)))
            if src and src in df.columns:
                return df[src]
     
        m = re.fullmatch(r"Temperature (\d+)m syn \[°C\]", col)
        if m:
            h = m.group(1)
            corrected_col = find_corrected_temperature_column(df, h)
            if corrected_col is not None:
                series = df[corrected_col].astype(float)
                if corrected_col.endswith("K"):
                    return series - 273.15
                return series
        # recomputed turbulence intensity
        m = re.fullmatch(r"Speed (\d+)m syn TI \[%\]", col)
        if m and f"new_TI_{m.group(1)}m" in df.columns:
            return df[f"new_TI_{m.group(1)}m"]
        # density
        m = re.fullmatch(r"Density (\d+)m \[kg/m³\]", col)
        if m:
            h = m.group(1)
            recomputed = f"recomputed_density_{h}m"
            density_col = f"Density {h}m [kg/m³]"
            if recomputed in df.columns:
                return df[recomputed]
            if density_col in df.columns:
                return df[density_col]

        # everything else
        return df[col] if col in df.columns else pd.Series([""] * len(df))

    out = pd.DataFrame({col: resolve(col) for col in out_columns})

    #mast file headers
    ensure_output_dir(config)
    out_path = config.path(f"final_timeseries_{mast_id}.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fout:
        for line in meta_lines:
            fout.write(line + "\n")
        fout.write(",".join(out_columns) + "\n")
        out.to_csv(fout, index=False, header=False)
    return out_path

#stage 8 plots
plot_c_raw = "blue"
plot_c_corrected = "orange"
plot_c_bestfit = "green"
plot_c_neutral = "gray"
plot_c_text = "black"
plot_c_axisgrid = "white"
plot_month_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def apply_plot_style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=plot_c_axisgrid, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("gray")
    ax.tick_params(labelsize=9, labelcolor=plot_c_text)

def run_stage8(config, targets=None, target_glob=None):
    ensure_output_dir(config)
    masts = discover_masts(config, targets=targets, target_glob=target_glob)
    windows_csv = config.path("window_diagnostics.csv")
    df_windows = pd.read_csv(windows_csv).sort_values("Rank").reset_index(drop=True)
    df_reference = load_reference(config.reference_csv)

    for mast in masts:
        print(f"Stage 8 diagnostic plots for mast {mast.mast_id}")
        df_original = load_mast_input(mast.raw_path)
        df_corrected = pd.read_csv(mast.best_window_path)

        speed_col_original = f"Speed {config.primary_speed_height}m syn [m/s]"
        speed_col_corrected = "best_window_corrected_speed"

        df_original["Month_int"] = df_original["Month"].astype(int)
        df_corrected["Month_int"] = df_corrected["Month"].astype(int)
        orig_monthly = df_original.groupby("Month_int")[speed_col_original].mean().reindex(range(1, 13))
        corr_monthly = df_corrected.groupby("Month_int")[speed_col_corrected].mean().reindex(range(1, 13))

        #before after monthly
        fig, ax = plt.subplots(figsize=(10, 5))
        apply_plot_style(ax)
        x = np.arange(12)
        width = 0.4
        bars1 = ax.bar(x - width / 2, orig_monthly, width, label="original", color=plot_c_raw, zorder=3, linewidth=0)
        bars2 = ax.bar(x + width / 2, corr_monthly, width, label="corrected best window", color=plot_c_corrected, zorder=3, linewidth=0)
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color=plot_c_text)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_month_abbr, fontsize=10, color=plot_c_text)
        ax.set_ylabel("mean wind speed m/s", fontsize=10, color=plot_c_text)
        ax.set_title(f"monthly wind speed before/after correction @ {config.primary_speed_height}m - mast {mast.mast_id}", fontsize=13, fontweight="bold", color=plot_c_text, pad=12)
        ax.legend(frameon=False, fontsize=9)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        fig.tight_layout()
        out_monthly = config.path(f"monthly_before_after_{mast.mast_id}.png")
        fig.savefig(out_monthly, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_monthly}")

    #mad plot
    n_windows = len(df_windows)
    if n_windows > 0:
        best_rank = df_windows["Rank"].iloc[0]
        best_label = df_windows["window"].iloc[0]
        bar_colors = [plot_c_bestfit if r == best_rank else plot_c_neutral for r in df_windows["Rank"]]
        fig_h = max(4, n_windows * 0.28)
        fig, ax = plt.subplots(figsize=(9, fig_h))
        apply_plot_style(ax)
        ax.grid(axis="x", color=plot_c_axisgrid, linewidth=0.8, zorder=0)
        ax.grid(axis="y", visible=False)
        y_pos = np.arange(n_windows)
        ax.barh(y_pos, df_windows["Combined_MAD"], color=bar_colors, zorder=3, height=0.7, linewidth=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_windows["window"], fontsize=8, color=plot_c_text)
        ax.invert_yaxis()
        ax.set_xlabel("combined MAD (mean |ratio - 1|)", fontsize=10, color=plot_c_text)
        ax.set_title(f"MAD score per window\nbest: {best_label}", fontsize=13, fontweight="bold", color=plot_c_text, pad=10)
        fig.tight_layout()
        out_mad = config.path("mad_bar_chart.png")
        fig.savefig(out_mad, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_mad}")

        #heatmap
        long_term_means = df_reference.groupby(df_reference.index.month)[wind_reference_col].mean().reindex(range(1, 13)).values
        heatmap_data = []
        window_labels = []
        for _, row in df_windows.iterrows():
            win_start = pd.Timestamp(row["Start"])
            win_end = pd.Timestamp(row["End"])
            window_labels.append(row["window"])
            subset = df_reference.loc[win_start:win_end, wind_reference_col].dropna()
            win_monthly = subset.groupby(subset.index.month).mean().reindex(range(1, 13)).values
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(win_monthly > 0, long_term_means / win_monthly, np.nan)
            ratios = np.clip(ratios, config.wind_ratio_cap_min, config.wind_ratio_cap_max)
            heatmap_data.append(ratios)

        heat_matrix = np.column_stack(heatmap_data)
        fig_w = max(8, n_windows * max(0.55, min(1.0, 9.0 / n_windows)) + 3)
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        norm = TwoSlopeNorm(vmin=config.wind_ratio_cap_min, vcenter=1.0, vmax=config.wind_ratio_cap_max)
        im = ax.imshow(heat_matrix, aspect="auto", cmap="RdYlGn", norm=norm, interpolation="nearest")
        ax.set_xticks(range(n_windows))
        ax.set_xticklabels(window_labels, rotation=55, ha="right", fontsize=max(5.5, 8 - n_windows // 8), color=plot_c_text)
        ax.set_yticks(range(12))
        ax.set_yticklabels(plot_month_abbr, fontsize=9, color=plot_c_text)
        ax.set_xlabel("window", fontsize=10, color=plot_c_text)
        ax.set_ylabel("month", fontsize=10, color=plot_c_text)
        ax.set_title("scaling ratio heatmap", fontsize=13, fontweight="bold", color=plot_c_text, pad=10)
        rect = Rectangle((-0.5, -0.5), 1, 12, linewidth=2, edgecolor=plot_c_bestfit, facecolor="none", zorder=5)
        ax.add_patch(rect)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("applied ratio", fontsize=9, color=plot_c_text)
        fig.tight_layout()
        out_heatmap = config.path("scaling_ratio_heatmap.png")
        fig.savefig(out_heatmap, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_heatmap}")

functions = {
    1: ("Reference climatology", stage1_climatology),
    2: ("Scale target wind speed", run_stage2),
    3: ("Rolling-window MAD best-window selection", run_stage3),
    4: ("Scale target temperature", run_stage4),
    5: ("Elevation back-calculation", run_stage5),
    6: ("Recompute turbulence intensity", run_stage6),
    7: ("Merge wind/TI and temperature outputs", run_stage7),
    8: ("Diagnostic plots", run_stage8),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run the revised MDM pipeline.")
    
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=list(functions.keys()),
        choices=list(functions.keys()),
    )
    
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--target-glob",
        default=None,
    )

    parser.add_argument(
        "--reference-csv",
        default=reference_csv_default,
    )

    parser.add_argument(
        "--output-dir",
        default=output_dir_default,
    )

    parser.add_argument(
        "--target-start",
        default=None,
    )

    parser.add_argument(
        "--target-end",
        default=None,
    )

    parser.add_argument(
        "--heights",
        nargs="+",
        type=int,
        default=default_heights,
    )

    parser.add_argument(
        "--primary-speed-height",
        type=int,
        default=primary_speed_height,
    )

    parser.add_argument(
        "--write-diagnostics",
        action="store_true",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = PipelineConfig(
        output_dir=args.output_dir,
        reference_csv=args.reference_csv,
        target_start=args.target_start,
        target_end=args.target_end,
        heights=args.heights,
        primary_speed_height=args.primary_speed_height,
        write_diagnostics=args.write_diagnostics,
    )

    # find masts
    masts = discover_masts(config, targets=args.targets, target_glob=args.target_glob)

    if args.target_start is None or args.target_end is None:
        inferred_start, inferred_end = infer_target_window_from_masts(masts)
        if args.target_start is None:
            config.target_start = inferred_start
        if args.target_end is None:
            config.target_end = inferred_end

    print(f"Target window: {config.target_start} -> {config.target_end}")

    # overlap
    overlap_start, overlap_end = compute_global_overlap(config, masts)
    config.overlap_start = overlap_start
    config.overlap_end = overlap_end
    print(f"Global overlap: {overlap_start} -> {overlap_end}")

    for stage_num in args.stages:
        # look up the stage name and function
        name, func = functions[stage_num]
        print(f"\n=== Stage {stage_num}: {name} ===")
       
       #progress checks
        if stage_num == 1:
            func(config)
        else:
            func(config, targets=args.targets, target_glob=args.target_glob)

if __name__ == "__main__":
    main()

