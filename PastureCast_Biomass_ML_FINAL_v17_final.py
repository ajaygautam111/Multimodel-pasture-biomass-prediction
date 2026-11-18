# # Pasture biomass modeling | Final v17
#
# This notebook provides the final manuscript-oriented analysis for pasture biomass estimation using canopy height, PlanetScope vegetation indices, phenological timing, and weather/soil-water predictors. Ridge regression, XGBoost, and ANN are evaluated under identical data partitions, with model-specific tuning and consistent validation diagnostics.
#


# ## Step 0 | Configuration
# Define the reproducible analysis environment: data/output paths, candidate predictors, model colors, random seed, validation settings, and figure aesthetics. All later steps use these shared settings.

from __future__ import annotations

import json
import hashlib
from decimal import Decimal
import logging
import gc
import math
import os
import platform
import random
import re
import string
import textwrap
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Markdown, display

from scipy.stats import randint, loguniform, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

# Do not suppress convergence warnings globally; fitting warnings are saved in the audit.
warnings.filterwarnings('ignore', category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

FAST_MODE = bool(int(os.environ.get("PASTURECAST_FAST_MODE", "0")))

CANDIDATE_DATA_PATHS = [
    Path(r"C:\Users\ag2c2\OneDrive - University of Missouri\RESEARCH\Testbed Paper\data\TB_24_25_VI_clean_data.csv"),
    Path("/mnt/data/TB_24_25_VI_clean_data.csv"),
    Path("TB_24_25_VI_clean_data.csv"),
]
DATA_PATH = Path(os.environ.get('PASTURECAST_DATA_PATH', str(next((p for p in CANDIDATE_DATA_PATHS if p.exists()), CANDIDATE_DATA_PATHS[0]))))

WINDOWS_OUTPUT_DIR = Path(r"C:\Users\ag2c2\OneDrive - University of Missouri\RESEARCH\Testbed Paper\data\manuscript_outputs_ann_v17_final")
OUT_DIR = Path(os.environ.get('PASTURECAST_OUT_DIR', str(WINDOWS_OUTPUT_DIR if os.name == 'nt' else Path.cwd() / 'manuscript_outputs_ann_v17_final')))
FIG_DIR = OUT_DIR / "figures"
PNG_FIG_DIR = OUT_DIR / "figurepng"
TABLE_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"
for directory in [OUT_DIR, FIG_DIR, PNG_FIG_DIR, TABLE_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TARGET = "Total Biomass (kg/ha)"
HEIGHT_FEATURES = ["PT_Height(mm)"]
VI_FEATURES = ["NDVI_mean", "GNDVI_mean", "EVI_mean", "SAVI_mean", "MSAVI_mean", "NDRE_mean", "CLRE_mean"]
PHENO_NUMERIC_FEATURES = ["Julian_Cos"]
PHENO_CATEGORICAL_FEATURES = ["season"]
WEATHER_FEATURES = [
    "precip_7d_sum_in", "et_7d_sum", "swb_7d_avg_pct", "tavg_7d_avg_F",
    "precip_14d_sum_in", "et_14d_sum", "swb_14d_avg_pct", "tavg_14d_avg_F",
    "precip_21d_sum_in", "et_21d_sum", "swb_21d_avg_pct", "tavg_21d_avg_F",
]
NUMERIC_CANDIDATES = HEIGHT_FEATURES + VI_FEATURES + PHENO_NUMERIC_FEATURES + WEATHER_FEATURES
CATEGORICAL_CANDIDATES = PHENO_CATEGORICAL_FEATURES

FEATURE_GROUPS = {TARGET: "Biomass"}
FEATURE_GROUPS.update({feature: "Canopy Height" for feature in HEIGHT_FEATURES})
FEATURE_GROUPS.update({feature: "PlanetScope VI" for feature in VI_FEATURES})
FEATURE_GROUPS.update({feature: "Phenological Timeline" for feature in PHENO_NUMERIC_FEATURES + PHENO_CATEGORICAL_FEATURES})
FEATURE_GROUPS.update({feature: "Weather and Soil Variables" for feature in WEATHER_FEATURES})

FEATURE_LABELS = {
    TARGET: "Total biomass (kg ha$^{-1}$)",
    "PT_Height(mm)": "Canopy height (mm)",
    "NDVI_mean": "NDVI (PlanetScope)",
    "GNDVI_mean": "GNDVI (PlanetScope)",
    "EVI_mean": "EVI (PlanetScope)",
    "SAVI_mean": "SAVI (PlanetScope)",
    "MSAVI_mean": "MSAVI (PlanetScope)",
    "NDRE_mean": "NDRE (PlanetScope)",
    "CLRE_mean": "CLRE (PlanetScope)",
    "Julian_Cos": "Cosine of Julian day",
    "season": "Season",
    "precip_7d_sum_in": "7-d precipitation (in)",
    "et_7d_sum": "7-d evapotranspiration",
    "swb_7d_avg_pct": "7-d soil water balance (%)",
    "tavg_7d_avg_F": "7-d mean air temperature (\u00b0F)",
    "precip_14d_sum_in": "14-d precipitation (in)",
    "et_14d_sum": "14-d evapotranspiration",
    "swb_14d_avg_pct": "14-d soil water balance (%)",
    "tavg_14d_avg_F": "14-d mean air temperature (\u00b0F)",
    "precip_21d_sum_in": "21-d precipitation (in)",
    "et_21d_sum": "21-d evapotranspiration",
    "swb_21d_avg_pct": "21-d soil water balance (%)",
    "tavg_21d_avg_F": "21-d mean air temperature (\u00b0F)",
    "Plot": "Plot",
    "Strip": "Strip",
}

MODEL_COLORS = {
    "Ridge": "#0072B2",
    "XGBoost": "#D55E00",
    "ANN": "#009E73",
}
MODEL_ORDER = ["Ridge", "XGBoost", "ANN"]
MODEL_DISPLAY_NAMES = {"Ridge": "Ridge regression", "XGBoost": "XGBoost", "ANN": "ANN"}
BASELINE_HEIGHT_NAME = "Height only linear regression"
BASELINE_STEPWISE_NAME = "Height + PlanetScope's VI linear stepwise regression"
BASELINE_COLORS = {BASELINE_HEIGHT_NAME: "#009E73", BASELINE_STEPWISE_NAME: "#0072B2"}

SEASON_COLORS = {
    "Spring": "#009E73",
    "Summer": "#F0E442",
    "Fall": "#8C510A",
    "Unknown": "#999999",
}
TREATMENT_COLORS = {
    "14": "#0072B2",
    "28": "#009E73",
    "42": "#D55E00",
    "Unknown": "#999999",
}
DATE_COLORS = ["#F0E442", "#009E73", "#8C510A", "#0072B2", "#CC79A7", "#D55E00"]
FEATURE_CLASS_COLORS = {
    "Biomass": "#000000",
    "Canopy Height": "#0072B2",
    "PlanetScope VI": "#009E73",
    "Phenological Timeline": "#CC79A7",
    "Weather and Soil Variables": "#E69F00",
}

FIGURE_STYLE = {
    "base_font_size": 17,
    "axis_label_size": 21,
    "tick_label_size": 17,
    "title_size": 22,
    "panel_letter_size": 23,
    "legend_font_size": 16,
    "legend_title_size": 17,
    "annotation_font_size": 16,
    "metric_box_size": 16,
    "save_dpi": 350 if FAST_MODE else 600,
    "scatter_size": 34,
    "scatter_alpha": 0.68,
    "line_width": 2.4,
    "fit_line_width": 2.6,
    "one_to_one_width": 2.1,
    "grid_alpha": 0.20,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "figure.dpi": 120,
    "savefig.dpi": FIGURE_STYLE["save_dpi"],
    "font.size": FIGURE_STYLE["base_font_size"],
    "axes.titlesize": FIGURE_STYLE["title_size"],
    "axes.labelsize": FIGURE_STYLE["axis_label_size"],
    "xtick.labelsize": FIGURE_STYLE["tick_label_size"],
    "ytick.labelsize": FIGURE_STYLE["tick_label_size"],
    "legend.fontsize": FIGURE_STYLE["legend_font_size"],
    "legend.title_fontsize": FIGURE_STYLE["legend_title_size"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": FIGURE_STYLE["grid_alpha"],
    "axes.titleweight": "normal",
})

CORRELATION_THRESHOLD = 0.90
VIF_THRESHOLD = 10.0
MIN_FEATURES_AFTER_SCREENING = 4
CV_FOLDS = 5
INNER_CV_FOLDS = 5 if not FAST_MODE else 2
TARGETED_DATE_HOLDOUT_DATES = ["2024-10-02", "2025-04-15", "2025-09-16"]
FORWARD_HOLDOUT_N_DATES = 3

RIDGE_ALPHA_GRID = np.logspace(-4, 5, 41) if not FAST_MODE else np.logspace(-2, 2, 5)
XGB_N_ITER = 60 if not FAST_MODE else 2
ANN_N_ITER = 30 if not FAST_MODE else 2
ANN_MAX_ITER = 1000 if not FAST_MODE else 120
SEARCH_N_JOBS = -1 if not FAST_MODE else 1
PERMUTATION_REPEATS = 8 if not FAST_MODE else 2
SHAP_SAMPLE_SIZE = 300 if not FAST_MODE else 60

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

print("PastureCast final version 15 configuration")
print("Data:", DATA_PATH)
print("Output:", OUT_DIR)
print("FAST_MODE:", FAST_MODE)

# Panel typography: sizes are in points and are editable in one place.
PANEL_STYLE = {'title': 26, 'label': 23, 'tick': 19, 'legend': 20,
               'value': 19, 'feature': 18, 'fold_legend': 16}
HEATMAP_STYLE = {'cell': 18, 'variable': 21, 'legend': 20}
# Correlation heatmap uses the exact previous seaborn diverging palette in STEP 3A.
FOLD_LEGEND_MAX_GROUPS = 6
SHOW_FIGURES = bool(int(os.environ.get('PASTURECAST_SHOW_FIGURES', '1')))
EXPORT_PNG = True
MAX_EXPORT_SIDE_PX = int(os.environ.get('PASTURECAST_MAX_EXPORT_SIDE_PX', '12000'))
STRIP_GROUP_COLUMN = 'Strip'  # Source design: 22 strip-position labels pooled across plots.
UNCERTAINTY_COVERAGE = 0.90
UNCERTAINTY_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
CALIBRATION_DATE_FRACTION = 0.20
RUN_SPLIT_CALIBRATION = True
PROTOCOL_ORDER = ['kfold_cv', 'leave_one_strip_out', 'leave_one_season_out',
                  'leave_one_treatment_out', 'targeted_date_holdout', 'forward_date_holdout']
PROTOCOL_LABELS = {'kfold_cv':'Five-fold CV', 'leave_one_strip_out':'Strip leave-out',
                  'leave_one_season_out':'Season leave-out', 'leave_one_treatment_out':'Treatment leave-out',
                  'targeted_date_holdout':'Selected-date holdout', 'forward_date_holdout':'Forward-date holdout'}
OUTPUT_LOG=[]; FIT_WARNINGS=[]; SCREENING_AUDITS=[]; PARAMETER_AUDIT=[]
SPLIT_REGISTRY={}; SPLIT_SUMMARIES={}


# ## Step 0.1 | Shared analysis functions
# Define reusable functions for labels, metrics, exports, preprocessing, model construction, and plotting. Centralizing these operations keeps model comparisons and manuscript figures internally consistent.

def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

def pretty_feature(name: object) -> str:
    name = str(name).replace("num__", "").replace("cat__", "")
    if name.startswith("season_"):
        return "Season: " + name.split("season_", 1)[1]
    return FEATURE_LABELS.get(name, name)

def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    OUTPUT_LOG.append({'type': 'table', 'path': str(path), 'rows': len(df)})
    return path

def save_figure(fig: plt.Figure, stem: str, show: bool = True):
    """Preserve the layout; bound raster size to avoid multi-gigabyte panels."""
    jpeg_path = FIG_DIR / f'{stem}.jpeg'
    png_path = PNG_FIG_DIR / f'{stem}.png'
    dpi = min(FIGURE_STYLE['save_dpi'], MAX_EXPORT_SIDE_PX / max(fig.get_size_inches()))
    fig.savefig(jpeg_path, dpi=dpi, bbox_inches='tight', facecolor='white',
                pil_kwargs={'quality': 97, 'subsampling': 0})
    if EXPORT_PNG:
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    OUTPUT_LOG.append({'type': 'figure', 'path': str(jpeg_path), 'effective_dpi': dpi})
    if show and SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return jpeg_path, png_path

def wrap_label(value: object, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))

def model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)

def model_color(model_name: str) -> str:
    return MODEL_COLORS.get(model_name, BASELINE_COLORS.get(model_name, "#777777"))

def format_group_label(protocol: str, group: object) -> str:
    label = str(group)
    if protocol == "leave_one_treatment_out":
        return f"{label} days"
    if protocol == "leave_one_strip_out":
        return f"Strip {label}"
    return label

def compact_feature_label(feature: object) -> str:
    mapping = {
        TARGET: "Biomass",
        "PT_Height(mm)": "Canopy height",
        "NDVI_mean": "NDVI",
        "GNDVI_mean": "GNDVI",
        "EVI_mean": "EVI",
        "SAVI_mean": "SAVI",
        "MSAVI_mean": "MSAVI",
        "NDRE_mean": "NDRE",
        "CLRE_mean": "CLRE",
        "Julian_Cos": "Julian cos",
        "precip_7d_sum_in": "Precip 7 d",
        "et_7d_sum": "ET 7 d",
        "swb_7d_avg_pct": "SWB 7 d",
        "tavg_7d_avg_F": "Tavg 7 d",
        "precip_14d_sum_in": "Precip 14 d",
        "et_14d_sum": "ET 14 d",
        "swb_14d_avg_pct": "SWB 14 d",
        "tavg_14d_avg_F": "Tavg 14 d",
        "precip_21d_sum_in": "Precip 21 d",
        "et_21d_sum": "ET 21 d",
        "swb_21d_avg_pct": "SWB 21 d",
        "tavg_21d_avg_F": "Tavg 21 d",
    }
    return mapping.get(str(feature), FEATURE_LABELS.get(str(feature), str(feature)))

def calibration_metrics(y_true, y_pred):
    """Return bounded reported R², RMSE, MAE, bias, and fitted-line terms.

    The displayed R² reproduces the earlier figures: squared Pearson
    correlation between observed and predicted values (0–1). It is NOT
    the predictive coefficient of determination 1-SSE/SST. RMSE, MAE
    and Bias are reported to assess absolute prediction error.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    n = len(y_true)
    result = dict(n=n, R2=np.nan, RMSE=np.nan, MAE=np.nan,
                  Bias=np.nan, slope=np.nan, intercept=np.nan)
    if n == 0:
        return result
    result.update(
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAE=float(mean_absolute_error(y_true, y_pred)),
        Bias=float(np.mean(y_pred - y_true)),
    )
    if n > 1 and np.std(y_true) > 0:
        if np.std(y_pred) > 0:
            result['R2'] = float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)
        line = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
        result.update(slope=float(line.coef_[0]), intercept=float(line.intercept_))
    return result
def metrics_row(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    model: str,
    protocol: str,
    fold: object,
    held_out_group: object,
    selected_features: Sequence[str],
) -> Dict[str, object]:
    metrics = calibration_metrics(y_true, y_pred)
    return {
        'model': model,
        'protocol': protocol,
        'fold': str(fold),
        'held_out_group': str(held_out_group),
        **metrics,
        'n_features': int(len(selected_features)),
        'selected_features': ' | '.join(map(str, selected_features)),
    }


def average_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    rows = []
    for (protocol, model), group in metrics_df.groupby(['protocol', 'model'], sort=False):
        row = {
            'protocol': protocol,
            'model': model,
            'n_repetitions': int(len(group)),
            'n_observations_total': int(group['n'].sum()),
        }
        for metric in ['R2', 'RMSE', 'MAE', 'Bias']:
            values = pd.to_numeric(group[metric], errors='coerce')
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_sd'] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
            row[f'{metric}_min'] = float(values.min())
            row[f'{metric}_max'] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def append_average_accuracy_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Append one arithmetic average row per protocol/model without treating it as an observation."""
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame() if metrics_df is None else metrics_df.copy()
    output = metrics_df.copy()
    average_rows = []
    for (protocol, model), group in metrics_df.groupby(['protocol', 'model'], sort=False):
        average_rows.append({
            'model': model,
            'protocol': protocol,
            'fold': 'Average accuracy',
            'held_out_group': 'Average accuracy',
            'n': int(pd.to_numeric(group['n'], errors='coerce').sum()),
            'R2': float(pd.to_numeric(group['R2'], errors='coerce').mean()),
            'RMSE': float(pd.to_numeric(group['RMSE'], errors='coerce').mean()),
            'MAE': float(pd.to_numeric(group['MAE'], errors='coerce').mean()),
            'Bias': float(pd.to_numeric(group['Bias'], errors='coerce').mean()),
            'slope': float(pd.to_numeric(group['slope'], errors='coerce').mean()),
            'intercept': float(pd.to_numeric(group['intercept'], errors='coerce').mean()),
            'n_features': float(pd.to_numeric(group['n_features'], errors='coerce').mean()),
            'selected_features': 'Average across repetitions',
        })
    return pd.concat([output, pd.DataFrame(average_rows)], ignore_index=True, sort=False)

def clean_numeric_matrix(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    available = [column for column in columns if column in data.columns]
    matrix = data[available].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    matrix = matrix.dropna(axis=1, how="all")
    matrix = matrix.loc[:, matrix.nunique(dropna=True) > 1]
    if matrix.empty:
        return matrix
    return matrix.fillna(matrix.median(numeric_only=True))


# ### Feature-screening and validation functions
# Define Pearson redundancy screening, VIF pruning, mutual-information scoring, preprocessing, and model-fitting helpers. Screening is applied to training data within validation workflows so that held-out observations do not determine the fitted predictor set.

def target_abs_pearson(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    y_numeric = pd.to_numeric(y, errors="coerce")
    output = {}
    for column in X.columns:
        x = pd.to_numeric(X[column], errors="coerce")
        valid = x.notna() & y_numeric.notna()
        if valid.sum() < 3 or x.loc[valid].nunique() < 2 or y_numeric.loc[valid].nunique() < 2:
            output[column] = 0.0
        else:
            value = x.loc[valid].corr(y_numeric.loc[valid], method="pearson")
            output[column] = 0.0 if pd.isna(value) else abs(float(value))
    return pd.Series(output, name="abs_pearson_with_biomass")

def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    if X.empty:
        return pd.DataFrame(columns=["feature", "VIF"])
    if X.shape[1] == 1:
        return pd.DataFrame({"feature": X.columns, "VIF": [1.0]})
    scaled = StandardScaler().fit_transform(X.values)
    rows = []
    for index, column in enumerate(X.columns):
        try:
            value = float(variance_inflation_factor(scaled, index))
        except Exception:
            value = np.inf
        rows.append({"feature": column, "VIF": value})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)

def pearson_prune(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = CORRELATION_THRESHOLD,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    if X.shape[1] <= 1:
        return list(X.columns), pd.DataFrame(), pd.DataFrame()
    association = target_abs_pearson(X, y)
    correlation = X.corr(method="pearson").abs()
    remaining = set(X.columns)
    removed = []
    pairs = []
    columns = list(correlation.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            value = correlation.loc[first, second]
            if pd.notna(value) and value >= threshold:
                pairs.append((first, second, float(value)))
    for first, second, value in sorted(pairs, key=lambda item: item[2], reverse=True):
        if first not in remaining or second not in remaining:
            continue
        if association.get(first, 0.0) >= association.get(second, 0.0):
            keep, drop = first, second
        else:
            keep, drop = second, first
        remaining.remove(drop)
        removed.append({
            "dropped_feature": drop,
            "retained_feature": keep,
            "abs_pearson_between_features": value,
            "dropped_abs_pearson_with_biomass": association.get(drop, np.nan),
            "retained_abs_pearson_with_biomass": association.get(keep, np.nan),
        })
    kept = [column for column in X.columns if column in remaining]
    pair_table = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "abs_pearson"])
    return kept, pd.DataFrame(removed), pair_table

def iterative_vif_prune(
    X: pd.DataFrame,
    threshold: float = VIF_THRESHOLD,
    minimum_features: int = MIN_FEATURES_AFTER_SCREENING,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    remaining = list(X.columns)
    steps = []
    while len(remaining) > max(1, minimum_features):
        vif = compute_vif_table(X[remaining])
        if vif.empty:
            break
        top = vif.iloc[0]
        top_value = float(top["VIF"])
        if np.isfinite(top_value) and top_value <= threshold:
            break
        drop = str(top["feature"])
        remaining.remove(drop)
        steps.append({
            "step": len(steps) + 1,
            "dropped_feature": drop,
            "VIF_at_drop": top_value,
            "n_features_remaining": len(remaining),
        })
    final_vif = compute_vif_table(X[remaining]) if remaining else pd.DataFrame(columns=["feature", "VIF"])
    return remaining, pd.DataFrame(steps), final_vif

def mutual_information_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    if X.empty:
        return pd.DataFrame(columns=["feature", "mutual_information"])
    y_numeric = pd.to_numeric(y, errors="coerce")
    valid = y_numeric.notna()
    values = mutual_info_regression(X.loc[valid], y_numeric.loc[valid], random_state=RANDOM_STATE)
    return pd.DataFrame({"feature": X.columns, "mutual_information": values}).sort_values(
        "mutual_information", ascending=False
    )

def screen_features(
    train_data: pd.DataFrame,
    display_outputs: bool = False,
    tag: str = "training",
) -> Dict[str, object]:
    numeric = clean_numeric_matrix(train_data, NUMERIC_CANDIDATES)
    y = pd.to_numeric(train_data[TARGET], errors="coerce")
    association = target_abs_pearson(numeric, y).reset_index().rename(columns={"index": "feature"})
    mutual_information = mutual_information_table(numeric, y)
    raw_vif = compute_vif_table(numeric).rename(columns={"VIF": "raw_VIF"})
    correlation_kept, correlation_removed, high_correlation_pairs = pearson_prune(numeric, y)
    correlation_matrix = numeric[correlation_kept] if correlation_kept else numeric.iloc[:, []]
    vif_kept, vif_steps, final_vif = iterative_vif_prune(correlation_matrix)

    if len(vif_kept) < MIN_FEATURES_AFTER_SCREENING and len(numeric.columns) >= MIN_FEATURES_AFTER_SCREENING:
        ranking = (
            association.merge(mutual_information, on="feature", how="outer")
            .fillna({"abs_pearson_with_biomass": 0.0, "mutual_information": 0.0})
        )
        ranking["combined_rank"] = (
            ranking["abs_pearson_with_biomass"].rank(ascending=False)
            + ranking["mutual_information"].rank(ascending=False)
        )
        fallback = ranking.sort_values("combined_rank")["feature"].head(MIN_FEATURES_AFTER_SCREENING).tolist()
        vif_kept = list(dict.fromkeys(vif_kept + fallback))

    # Recompute the actual final VIF after any minimum-count recovery.
    final_vif = compute_vif_table(numeric[vif_kept]) if vif_kept else pd.DataFrame(columns=['feature','VIF'])
    selected_categorical = [
        column
        for column in CATEGORICAL_CANDIDATES
        if column in train_data.columns and train_data[column].dropna().nunique() > 1
    ]
    selected_features = list(dict.fromkeys(vif_kept + selected_categorical))

    catalog = pd.DataFrame({"feature": list(numeric.columns) + selected_categorical})
    catalog["feature_type"] = np.where(catalog["feature"].isin(numeric.columns), "numeric", "categorical")
    catalog = catalog.merge(association, on="feature", how="left")
    catalog = catalog.merge(mutual_information, on="feature", how="left")
    catalog = catalog.merge(raw_vif, on="feature", how="left")
    catalog = catalog.merge(final_vif.rename(columns={"VIF": "final_VIF"}), on="feature", how="left")
    catalog["kept_after_pearson_pruning"] = catalog["feature"].isin(correlation_kept) | catalog["feature"].isin(selected_categorical)
    catalog["kept_after_vif_pruning"] = catalog["feature"].isin(vif_kept) | catalog["feature"].isin(selected_categorical)
    catalog["selected_for_model"] = catalog["feature"].isin(selected_features)
    catalog["final_VIF_above_threshold"] = catalog["final_VIF"].gt(VIF_THRESHOLD)
    catalog["feature_label"] = catalog["feature"].map(lambda item: FEATURE_LABELS.get(item, item))
    catalog = catalog.sort_values(
        ["selected_for_model", "feature_type", "mutual_information", "abs_pearson_with_biomass"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)

    result = {
        "selected_features": selected_features,
        "selected_numeric": [feature for feature in selected_features if feature != "season"],
        "selected_categorical": [feature for feature in selected_features if feature == "season"],
        "feature_catalog": catalog,
        "raw_vif": raw_vif,
        "mutual_information": mutual_information,
        "correlation_removed": correlation_removed,
        "high_correlation_pairs": high_correlation_pairs,
        "vif_steps": vif_steps,
        "final_vif": final_vif,
    }
    if display_outputs:
        display(Markdown(f"### Feature screening: {tag}"))
        display(Markdown(
            f"Candidate numeric predictors: **{len(numeric.columns)}**. "
            f"Selected predictors: **{len(selected_features)}**."
        ))
        display(catalog.round(4))
    return result

def make_preprocessor(selected_features: Sequence[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric = [feature for feature in selected_features if feature != "season"]
    categorical = [feature for feature in selected_features if feature == "season"]
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)),
            ]),
            categorical,
        ))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    return preprocessor, numeric, categorical

def build_model_pipeline(
    model_name: str,
    selected_features: Sequence[str],
    best_parameters: Optional[Dict[str, object]] = None,
) -> Pipeline:
    best_parameters = dict(best_parameters or {})
    preprocessor, _, _ = make_preprocessor(selected_features)
    if model_name == "Ridge":
        estimator = Ridge(alpha=float(best_parameters.get("alpha", 1.0)))
    elif model_name == "XGBoost":
        defaults = {
            "n_estimators": 350,
            "max_depth": 4,
            "learning_rate": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2,
            "gamma": 0.0,
            "reg_alpha": 0.01,
            "reg_lambda": 2.0,
        }
        defaults.update(best_parameters)
        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            **defaults,
        )
    elif model_name == "ANN":
        defaults = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "batch_size": 32,
            "learning_rate": "constant",
        }
        defaults.update(best_parameters)
        regressor = MLPRegressor(
            random_state=RANDOM_STATE,
            solver="adam",
            max_iter=ANN_MAX_ITER,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=35,
            **defaults,
        )
        estimator = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

def fit_predict_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_name: str,
    best_parameters: Dict[str, Dict[str, object]],
) -> Tuple[np.ndarray, Pipeline, Dict[str, object]]:
    screening = screen_features(train_data, display_outputs=False)
    selected = screening["selected_features"]
    pipeline = build_model_pipeline(model_name, selected, best_parameters.get(model_name, {}))
    pipeline.fit(train_data[selected], train_data[TARGET])
    predictions = pipeline.predict(test_data[selected])
    return np.asarray(predictions, dtype=float), pipeline, screening

def run_validation_protocol(data, splits, protocol, best_parameters, store_models=False):
    if protocol not in SPLIT_REGISTRY:
        splits = register_splits(data, splits, protocol)
    else:
        splits = SPLIT_REGISTRY[protocol]
    metric_rows, prediction_frames, fitted_models = [], [], {}
    for fold_number, (train_index, test_index, group_label) in enumerate(splits, 1):
        train_data, test_data = data.iloc[train_index].copy(), data.iloc[test_index].copy()
        screening = screen_features(train_data, display_outputs=False)
        selected = screening['selected_features']
        selection = screening['feature_catalog'].copy()
        selection['protocol'], selection['fold'] = protocol, str(fold_number)
        SCREENING_AUDITS.append(selection)
        for model_name in MODEL_ORDER:
            fitted = build_model_pipeline(model_name, selected, best_parameters[model_name])
            fitted = fit_with_audit(fitted, train_data, selected, model_name, protocol, fold_number)
            predicted = np.asarray(fitted.predict(test_data[selected]), dtype=float)
            metrics = metrics_row(test_data[TARGET], predicted, model_name, protocol,
                                  fold_number, group_label, selected)
            train_metrics = calibration_metrics(train_data[TARGET], fitted.predict(train_data[selected]))
            metrics.update(n_train=len(train_data), train_R2=train_metrics['R2'],
                           train_RMSE=train_metrics['RMSE'], train_MAE=train_metrics['MAE'],
                           RMSE_gap=metrics['RMSE']-train_metrics['RMSE'],
                           parameter_hash=parameter_signature(best_parameters[model_name]))
            metric_rows.append(metrics)
            frame = test_data[['row_id', 'sample_id', 'Date', 'season', 'Treatment_days', 'Plot', 'Strip']].copy()
            frame['Date'] = frame['Date'].dt.strftime('%Y-%m-%d')
            frame['observed'], frame['predicted'] = test_data[TARGET].to_numpy(float), predicted
            frame['model'], frame['protocol'], frame['fold'] = model_name, protocol, str(fold_number)
            frame['held_out_group'] = str(group_label)
            prediction_frames.append(frame)
            for name, value in best_parameters[model_name].items():
                PARAMETER_AUDIT.append(dict(protocol=protocol, fold=str(fold_number), model=model_name,
                                           parameter=name, value=decimal_text(value),
                                           parameter_hash=parameter_signature(best_parameters[model_name])))
            if store_models:
                fitted_models[(model_name, str(fold_number))] = fitted
    predictions = pd.concat(prediction_frames, ignore_index=True)
    for fold, subset in predictions.groupby('fold'):
        ids = [tuple(subset.loc[subset['model'].eq(m), 'sample_id']) for m in MODEL_ORDER]
        if not all(v == ids[0] for v in ids):
            raise AssertionError(f'Different test rows across models in {protocol}, fold {fold}.')
    return pd.DataFrame(metric_rows), predictions, fitted_models

def stepwise_aic_fit(
    train_data: pd.DataFrame,
    candidate_features: Sequence[str],
    forced_features: Sequence[str],
) -> Dict[str, object]:
    candidates = [feature for feature in candidate_features if feature in train_data.columns]
    forced = [feature for feature in forced_features if feature in candidates]
    X = train_data[candidates].apply(pd.to_numeric, errors="coerce")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    y = pd.to_numeric(train_data[TARGET], errors="coerce")
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    selected = list(forced)

    def fit_aic(features: Sequence[str]):
        design = sm.add_constant(X[list(features)], has_constant="add")
        model = sm.OLS(y, design).fit()
        return model, float(model.aic)

    current_model, current_aic = fit_aic(selected)
    changed = True
    while changed:
        changed = False
        excluded = [feature for feature in candidates if feature not in selected]
        forward = []
        for feature in excluded:
            model, aic = fit_aic(selected + [feature])
            forward.append((aic, feature, model))
        if forward:
            best_aic, best_feature, best_model = min(forward, key=lambda item: item[0])
            if best_aic + 1e-8 < current_aic:
                selected.append(best_feature)
                current_model = best_model
                current_aic = best_aic
                changed = True
        removable = [feature for feature in selected if feature not in forced]
        backward = []
        for feature in removable:
            reduced = [item for item in selected if item != feature]
            model, aic = fit_aic(reduced)
            backward.append((aic, feature, model, reduced))
        if backward:
            best_aic, drop_feature, best_model, reduced = min(backward, key=lambda item: item[0])
            if best_aic + 1e-8 < current_aic:
                selected = reduced
                current_model = best_model
                current_aic = best_aic
                changed = True
    return {
        "selected_features": selected,
        "model": current_model,
        "medians": medians,
        "aic": current_aic,
    }

def stepwise_predict(fit: Dict[str, object], test_data: pd.DataFrame) -> np.ndarray:
    selected = fit["selected_features"]
    X = test_data[selected].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(fit["medians"])
    design = sm.add_constant(X, has_constant="add")
    return np.asarray(fit["model"].predict(design), dtype=float)

def group_color_map(values, protocol):
    labels = [str(value) for value in values]
    if protocol == 'leave_one_season_out':
        return {label: SEASON_COLORS.get(label, '#999999') for label in labels}
    if protocol == 'leave_one_treatment_out':
        return {label: TREATMENT_COLORS.get(label, '#999999') for label in labels}
    return dict(zip(labels, distinct_colors(len(labels))))

def common_scatter_limits(predictions):
    values = np.r_[predictions['observed'].to_numpy(float), predictions['predicted'].to_numpy(float)]
    values = values[np.isfinite(values)]
    lower, upper = float(values.min()), float(values.max())
    pad = max((upper-lower)*0.05, 100.0)
    return (0.0 if lower >= 0 else lower-pad), upper+pad


# ### Shared plotting functions
# Define publication-style observed-versus-predicted, residual, fold, and stability figures. The same metrics, labels, colors, and axis conventions are used across all three model families.

def add_scatter_content(
    ax: plt.Axes,
    data: pd.DataFrame,
    color: str,
    title: str,
    metrics_box: bool = True,
    observation_label: str = "Observations",
    limits: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    metrics = calibration_metrics(data["observed"], data["predicted"])
    ax.scatter(
        data["observed"], data["predicted"],
        s=FIGURE_STYLE["scatter_size"], alpha=FIGURE_STYLE["scatter_alpha"],
        color=color, edgecolors="none", label=observation_label,
    )
    lower, upper = limits if limits is not None else common_scatter_limits(data)
    ax.plot(
        [lower, upper], [lower, upper], linestyle=":", color="black",
        linewidth=FIGURE_STYLE["one_to_one_width"], label="1:1 line",
    )
    x_line = np.linspace(lower, upper, 200)
    ax.plot(
        x_line, metrics["slope"] * x_line + metrics["intercept"],
        color="#B2182B", linewidth=FIGURE_STYLE["fit_line_width"], label="Fitted line",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Observed biomass (kg ha$^{-1}$)", fontsize=FIGURE_STYLE["axis_label_size"])
    ax.set_ylabel("Predicted biomass (kg ha$^{-1}$)", fontsize=FIGURE_STYLE["axis_label_size"])
    ax.set_title(title, fontsize=FIGURE_STYLE["title_size"], pad=12)
    ax.tick_params(axis="both", labelsize=FIGURE_STYLE["tick_label_size"])
    ax.tick_params(axis="y", labelrotation=0)
    if metrics_box:
        text = (
            f"n = {metrics['n']:,}\n"
            f"R² = {metrics['R2']:.2f}\n"
            f"RMSE = {metrics['RMSE']:,.2f}\n"
            f"MAE = {metrics['MAE']:,.2f}\n"
            f"Bias = {metrics['Bias']:,.2f}"
        )
        ax.text(
            0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=FIGURE_STYLE["metric_box_size"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BDBDBD", alpha=0.94),
        )
    ax.legend(
        loc="lower right", frameon=True, framealpha=0.94, edgecolor="#BDBDBD",
        fontsize=FIGURE_STYLE["legend_font_size"],
    )
    return metrics

def plot_single_repetition_scatter(
    data: pd.DataFrame,
    title: str,
    color: str,
    stem: str,
    observation_label: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    label = observation_label or str(data["held_out_group"].iloc[0])
    add_scatter_content(ax, data, color, title, observation_label=label)
    fig.tight_layout()
    save_figure(fig, stem)

def plot_repetition_grid(
    predictions: pd.DataFrame,
    model_name: str,
    protocol: str,
    stem: str,
    ncols_override: Optional[int] = None,
) -> None:
    data = predictions[(predictions["model"] == model_name) & (predictions["protocol"] == protocol)].copy()
    groups = list(dict.fromkeys(data["held_out_group"].astype(str).tolist()))
    colors = group_color_map(groups, protocol)
    n_groups = len(groups)
    if ncols_override is not None:
        ncols = min(ncols_override, max(1, n_groups))
    elif n_groups in {3, 5}:
        ncols = n_groups
    elif protocol == "leave_one_strip_out":
        ncols = 6
    else:
        ncols = min(4, max(1, n_groups))
    nrows = int(math.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.9 * ncols, 5.8 * nrows), squeeze=False)
    limits = common_scatter_limits(data)
    for index, group in enumerate(groups):
        ax = axes.flat[index]
        subset = data[data["held_out_group"].astype(str) == group]
        group_label = format_group_label(protocol, group)
        add_scatter_content(
            ax, subset, colors[group], group_label, metrics_box=True,
            observation_label=group_label, limits=limits,
        )
    for index in range(n_groups, nrows * ncols):
        axes.flat[index].axis("off")
    fig.subplots_adjust(wspace=0.30, hspace=0.35, top=0.96, bottom=0.08, left=0.06, right=0.99)
    save_figure(fig, stem)

def plot_collective_protocol_matrix(
    predictions: pd.DataFrame,
    protocol: str,
    stem: str,
    model_order: Sequence[str] = MODEL_ORDER,
) -> None:
    data = predictions[predictions["protocol"] == protocol].copy()
    groups = list(dict.fromkeys(data["held_out_group"].astype(str).tolist()))
    colors = group_color_map(groups, protocol)
    nrows = len(model_order)
    ncols = len(groups)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.9 * ncols, 5.6 * nrows),
        squeeze=False,
    )
    limits = common_scatter_limits(data)

    for row_index, model_name in enumerate(model_order):
        model_data = data[data["model"] == model_name]

        for column_index, group in enumerate(groups):
            ax = axes[row_index, column_index]
            subset = model_data[model_data["held_out_group"].astype(str) == group]
            group_label = format_group_label(protocol, group)

            add_scatter_content(
                ax,
                subset,
                colors[group],
                group_label,
                metrics_box=True,
                observation_label=group_label,
                limits=limits,
            )

            # Model names are presented once at the left edge of each row,
            # rather than repeated in every x-axis label.
            ax.set_xlabel(
                "Observed biomass (kg ha$^{-1}$)",
                fontsize=FIGURE_STYLE["axis_label_size"] - 1,
            )

        axes[row_index, 0].text(
            -0.43,
            0.50,
            model_display_name(model_name),
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FIGURE_STYLE["title_size"] + 1,
            fontweight="bold",
            clip_on=False,
        )

    matrix_hspace = 0.58 if nrows <= 2 else 0.46
    matrix_bottom = 0.085 if nrows <= 2 else 0.06
    fig.subplots_adjust(
        wspace=0.30,
        hspace=matrix_hspace,
        top=0.97,
        bottom=matrix_bottom,
        left=0.115,
        right=0.995,
    )
    save_figure(fig, stem)

def plot_fold_specific_lines_panel(predictions, protocol, stem, model_order=MODEL_ORDER):
    data = predictions.loc[predictions['protocol'].eq(protocol)].copy()
    groups = list(dict.fromkeys(data['held_out_group'].astype(str)))
    colors = group_color_map(groups, protocol)
    # No crowded 22-line legend: groups are divided into matched pages for each model.
    pages = [groups[i:i+FOLD_LEGEND_MAX_GROUPS] for i in range(0, len(groups), FOLD_LEGEND_MAX_GROUPS)]
    lo, hi = common_scatter_limits(data)
    for page, group_page in enumerate(pages, 1):
        fig, axes = plt.subplots(1, len(model_order), figsize=(9.4*len(model_order), 9.2), squeeze=False)
        for ax, model in zip(axes.ravel(), model_order):
            model_data = data.loc[data['model'].eq(model)]
            handles, labels = [], []
            for group in group_page:
                sub = model_data.loc[model_data['held_out_group'].astype(str).eq(group)]
                if sub.empty:
                    continue
                metric = calibration_metrics(sub['observed'], sub['predicted'])
                color, label = colors[group], format_group_label(protocol, group)
                ax.scatter(sub['observed'], sub['predicted'], s=32, alpha=0.56, color=color, edgecolors='none')
                grid = np.linspace(sub['observed'].min(), sub['observed'].max(), 100)
                if np.isfinite(metric['slope']):
                    ax.plot(grid, metric['slope']*grid+metric['intercept'], color=color, lw=2.7)
                handles.append(Line2D([0], [0], marker='o', color=color, lw=2.7, markersize=7))
                labels.append(f"{label} (n={metric['n']}): R² = {metric['R2']:.2f}, "
                              f"RMSE={metric['RMSE']:,.0f}\nMAE={metric['MAE']:,.0f}; Bias={metric['Bias']:,.0f}")
            ax.plot([lo,hi],[lo,hi],color='black',ls=':',lw=2.2)
            handles.append(Line2D([0],[0],color='black',ls=':',lw=2.2));labels.append('1:1 line')
            ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed biomass (kg ha$^{-1}$)',
                   ylabel='Predicted biomass (kg ha$^{-1}$)')
            ax.set_title(model_display_name(model), fontsize=PANEL_STYLE['title'],pad=12)
            ax.tick_params(labelsize=PANEL_STYLE['tick'])
            ax.xaxis.label.set_size(PANEL_STYLE['label']);ax.yaxis.label.set_size(PANEL_STYLE['label'])
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.17), ncol=1,
                      fontsize=PANEL_STYLE['fold_legend'],frameon=True,edgecolor='#BDBDBD',
                      handlelength=2.2,labelspacing=0.5,borderpad=0.7)
        fig.subplots_adjust(left=0.065,right=0.99,top=0.93,bottom=0.43,wspace=0.28)
        suffix = '' if len(pages)==1 else f'_page_{page:02d}'
        save_figure(fig,stem+suffix)

def plot_average_stability(
    metrics: pd.DataFrame,
    protocol: str,
    stem: str,
    model_order: Sequence[str] = MODEL_ORDER,
) -> pd.DataFrame:
    subset = metrics[metrics["protocol"] == protocol].copy()
    summary = average_metrics_table(subset)
    fig, axes = plt.subplots(1, 4, figsize=(28.5, 8.5))
    metric_specs = [
        ("R2", "R²", "R²"),
        ("RMSE", "RMSE", "RMSE (kg ha$^{-1}$)"),
        ("MAE", "MAE", "MAE (kg ha$^{-1}$)"),
        ("Bias", "Bias", "Bias (kg ha$^{-1}$)"),
    ]

    for panel_index, (metric, title, ylabel) in enumerate(metric_specs):
        ax = axes[panel_index]
        models = [model for model in model_order if model in summary["model"].tolist()]
        x = np.arange(len(models), dtype=float)
        means = [float(summary.loc[summary["model"] == model, f"{metric}_mean"].iloc[0]) for model in models]
        minimums = [float(summary.loc[summary["model"] == model, f"{metric}_min"].iloc[0]) for model in models]
        maximums = [float(summary.loc[summary["model"] == model, f"{metric}_max"].iloc[0]) for model in models]
        colors = [model_color(model) for model in models]

        ax.bar(x, means, width=0.68, color=colors, alpha=0.90)

        for position, minimum, maximum in zip(x, minimums, maximums):
            ax.vlines(position, minimum, maximum, colors="black", linestyles=":", linewidth=2.2, zorder=5)
            ax.hlines([minimum, maximum], position - 0.11, position + 0.11, colors="black", linewidth=1.6, zorder=5)

        finite_values = np.asarray(
            [value for value in means + minimums + maximums if np.isfinite(value)],
            dtype=float,
        )
        span = float(np.nanmax(finite_values) - np.nanmin(finite_values)) if len(finite_values) else 1.0
        label_offset = max(span * 0.012, 0.012 if metric == "R2" else 4.0)
        for position, mean in zip(x, means):   
            vertical = mean + label_offset if mean >= 0 else mean - label_offset
            ax.text(
                position + 0.18,
                vertical,
                f"{mean:.2f}" if metric == "R2" else f"{mean:,.0f}",
                ha="left",
                va="bottom" if mean >= 0 else "top",
                fontsize=24,
                fontweight="normal",
                clip_on=False,
            )



        
        ax.set_xticks(x)
        ax.set_xticklabels(
            [wrap_label(model_display_name(model), 17) for model in models],
            rotation=0,
            ha="center",
            fontsize=17.5,
        )
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_title(title, fontsize=22, pad=16)
        ax.tick_params(axis="y", labelsize=17.5, labelrotation=0)

        if metric == "R2":
            ax.set_ylim(0.0, 1.0)
        elif finite_values.size:
            pad = max(span * 0.14, 20.0)
            ax.set_ylim(float(np.nanmin(finite_values)) - pad, float(np.nanmax(finite_values)) + pad)

        if metric == "Bias":
            ax.axhline(0, color="black", linestyle=":", linewidth=1.5)

    fig.subplots_adjust(wspace=0.35, top=0.89, bottom=0.24, left=0.06, right=0.992)
    save_figure(fig, stem)
    return summary

def plot_all_protocol_average_stability(metrics, protocol_order, protocol_labels, stem):
    """Previous-version manuscript layout: six protocol rows × four metric columns."""
    summary = average_metrics_table(metrics.loc[metrics['protocol'].isin(protocol_order)])
    specs = [
        ('R2', 'R²'),
        ('RMSE', 'RMSE (kg ha$^{-1}$)'),
        ('MAE', 'MAE (kg ha$^{-1}$)'),
        ('Bias', 'Bias (kg ha$^{-1}$)'),
    ]

    fig, axes = plt.subplots(
        len(protocol_order), 4,
        figsize=(27.5, 4.65 * len(protocol_order)),
        squeeze=False,
    )

    for row_index, protocol in enumerate(protocol_order):
        subset = (
            summary.loc[summary['protocol'].eq(protocol)]
            .set_index('model')
            .reindex(MODEL_ORDER)
        )

        for col_index, (metric, ylabel) in enumerate(specs):
            ax = axes[row_index, col_index]
            x = np.arange(len(MODEL_ORDER), dtype=float)
            means = subset[f'{metric}_mean'].to_numpy(float)
            mins = subset[f'{metric}_min'].to_numpy(float)
            maxs = subset[f'{metric}_max'].to_numpy(float)

            bars = ax.bar(
                x, means, width=0.62,
                color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                alpha=0.92,
            )

            finite = np.r_[means, mins, maxs]
            finite = finite[np.isfinite(finite)]

            for xx, bar, mean, low, high in zip(x, bars, means, mins, maxs):
                if np.isfinite(low) and np.isfinite(high):
                    ax.vlines(xx, low, high, color='black', ls=':', lw=2.0, zorder=4)
                    ax.hlines([low, high], xx - 0.09, xx + 0.09, color='black', lw=1.4, zorder=4)
                if np.isfinite(mean):
                    label = f'{mean:.2f}' if metric == 'R2' else f'{0.0 if abs(mean) < .5 else mean:,.0f}'
                    ax.annotate(
                        label,
                        (bar.get_x() + bar.get_width()/2, mean),
                        xytext=(10, 8 if mean >= 0 else -8),
                        textcoords='offset points',
                        ha='left',
                        va='bottom' if mean >= 0 else 'top',
                        fontsize=23,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, pad=0.35),
                    )

            # Metric-specific scaling reproduces the compact previous figure while preserving all data.
            if metric == 'R2':
                ax.set_ylim(0.0, 1.0)
            elif finite.size:
                fmin, fmax = float(np.nanmin(finite)), float(np.nanmax(finite))
                span = max(fmax - fmin, 1.0)
                if metric in {'RMSE', 'MAE'}:
                    lower = max(0.0, fmin - 0.25 * span)
                    upper = fmax + 0.25 * span
                else:  # Bias
                    lower = min(0.0, fmin) - 0.18 * span
                    upper = max(0.0, fmax) + 0.18 * span
                if np.isclose(lower, upper):
                    upper = lower + 1.0
                ax.set_ylim(lower, upper)

            ax.set_xlim(-0.52, 2.55)
            ax.set_xticks(x)
            ax.set_xticklabels(['Ridge regression', 'XGBoost', 'ANN'], fontsize=19)
            ax.set_ylabel(ylabel, fontsize=23)
            ax.tick_params(axis='y', labelsize=19, labelrotation=0)

            if row_index == 0:
                ax.set_title('R²' if metric == 'R2' else metric, fontsize=24, pad=18, fontweight='normal')

            if col_index == 0:
                ax.text(
                    -0.29, 0.50,
                    protocol_labels[protocol].replace(' ', '\n', 1),
                    transform=ax.transAxes,
                    rotation=90,
                    ha='center', va='center',
                    fontsize=21,
                    fontweight='bold',
                )

            if metric == 'Bias':
                ax.axhline(0, color='black', ls=':', lw=1.15)

    # One-line legend above all metric titles, matching the previous manuscript layout.
    fig.legend(
        handles=[Patch(facecolor=MODEL_COLORS[m], label=model_display_name(m)) for m in MODEL_ORDER],
        loc='upper center',
        bbox_to_anchor=(0.53, 1.018),
        ncol=3,
        fontsize=19,
        frameon=True,
        edgecolor='#BDBDBD',
        borderpad=0.45,
        columnspacing=2.0,
        handlelength=1.8,
    )

    fig.subplots_adjust(
        left=0.11, right=0.995,
        bottom=0.035, top=0.945,
        wspace=0.31, hspace=0.43,
    )
    save_figure(fig, stem)
    return summary

def decimal_text(value):
    """Shortest stored decimal representation, without exponent notation."""
    from decimal import Decimal
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(Decimal(str(float(value))), 'f') if np.isfinite(value) else str(value)
    if isinstance(value, (tuple, list)):
        return '(' + ', '.join(decimal_text(v) for v in value) + ')'
    return str(value)

def plain_value(value):
    if isinstance(value, dict):
        return {str(k): plain_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def parameter_signature(parameters):
    payload = json.dumps(plain_value(parameters), sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

def distinct_colors(n):
    # Discrete, distinguishable colors first; additional strip groups use tab20b.
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
               '#56B4E9', '#332288', '#882255', '#44AA99', '#AA4499']
    import matplotlib.colors as mcolors
    palette += [mcolors.to_hex(plt.get_cmap('tab20b')(i)) for i in range(20)]
    return [palette[i % len(palette)] for i in range(n)]

def draw_boxplot(data, x, y, order, palette, ax, width=0.62):
    arrays = [pd.to_numeric(data.loc[data[x].astype(str) == str(g), y], errors='coerce').dropna()
              for g in order]
    result = ax.boxplot(arrays, positions=np.arange(len(order)), widths=width, patch_artist=True,
                        medianprops={'color': '#303030', 'linewidth': 1.4},
                        boxprops={'edgecolor': '#555555', 'linewidth': 1.0},
                        whiskerprops={'color': '#555555', 'linewidth': 1.0},
                        capprops={'color': '#555555', 'linewidth': 1.0},
                        flierprops={'marker': 'd', 'markersize': 3.5, 'markerfacecolor': '#666666',
                                    'markeredgecolor': '#666666', 'alpha': 0.55})
    for box, color in zip(result['boxes'], palette):
        box.set_facecolor(color)
        box.set_alpha(0.83)
    ax.set_xticks(np.arange(len(order)), [str(g) for g in order])
    ax.grid(axis='x', visible=False)
    return result

def register_splits(data, splits, protocol, purpose='evaluation'):
    """Persist exact membership once, then share the same list across algorithms."""
    saved, summary, memberships = [], [], []
    for fold, (train, test, group) in enumerate(splits, 1):
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        if not len(train) or not len(test):
            raise ValueError(f'{protocol}: empty partition in {group}.')
        if len(set(train) & set(test)):
            raise AssertionError(f'{protocol}: train/test row overlap in {group}.')
        tr, te = data.iloc[train], data.iloc[test]
        token = '|'.join(tr['sample_id']) + '||' + '|'.join(te['sample_id'])
        split_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        summary.append(dict(protocol=protocol, fold=str(fold), held_out_group=str(group),
                            purpose=purpose, n_train=len(train), n_test=len(test),
                            n_train_dates=tr['Date'].nunique(), n_test_dates=te['Date'].nunique(),
                            overlapping_dates=len(set(tr['Date']) & set(te['Date'])),
                            train_date_min=tr['Date'].min().date(), train_date_max=tr['Date'].max().date(),
                            test_date_min=te['Date'].min().date(), test_date_max=te['Date'].max().date(),
                            split_hash=split_hash))
        for role, part in [('training', tr), ('validation' if purpose == 'tuning' else 'testing', te)]:
            member = part[['row_id', 'sample_id', 'Date', 'season', 'Treatment_days', 'Plot', 'Strip']].copy()
            member.insert(0, 'role', role)
            member.insert(0, 'fold', str(fold))
            member.insert(0, 'protocol', protocol)
            member['split_hash'] = split_hash
            memberships.append(member)
        saved.append((train, test, group))
    save_table(pd.DataFrame(summary), f'00_{protocol}_split_summary.csv')
    save_table(pd.concat(memberships, ignore_index=True), f'00_{protocol}_sample_membership.csv')
    SPLIT_REGISTRY[protocol] = saved
    SPLIT_SUMMARIES[protocol] = pd.DataFrame(summary)
    return saved

def fit_with_audit(pipeline, train, features, model, protocol, fold):
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pipeline.fit(train[features], train[TARGET])
    for warning in caught:
        FIT_WARNINGS.append(dict(protocol=protocol, model=model, fold=str(fold),
                                 warning_type=warning.category.__name__, message=str(warning.message)))
    return pipeline


# ## Step 1 | Load and verify the modeling dataset
# Read the cleaned strip-date dataset, standardize identifiers and dates, and remove `Dry Matter %` and `SRre_mean` before any analysis. Required response, predictor, and grouping fields are then checked for completeness.

# Load once. Remove excluded columns BEFORE audits, descriptions, correlations, or selection.
if not DATA_PATH.exists():
    raise FileNotFoundError(f'Set DATA_PATH to the current cleaned modeling CSV. Not found: {DATA_PATH}')
raw_data = pd.read_csv(DATA_PATH)
input_rows = len(raw_data)
normalized = lambda name: re.sub(r'[^a-z0-9]', '', str(name).lower())
excluded_names = {'drymatter', 'drymatterpct', 'drymatterpercent', 'dmpercent', 'dmpct',
                  'drymatterpercentage', 'srre', 'srremean'}
removed_columns = [c for c in raw_data if normalized(c) in excluded_names]
raw_data = raw_data.drop(columns=removed_columns)
# Exclusion names are recorded only in the run manifest, not in analytical tables/plots.
raw_shape = raw_data.shape
empty_export_columns = [c for c in raw_data if str(c).startswith('Unnamed:')]
data = raw_data.drop(columns=empty_export_columns, errors='ignore').copy()
required_columns = [TARGET, 'PT_Height(mm)', 'Date', 'Plot', 'Strip']
missing_required = [c for c in required_columns if c not in data]
if missing_required:
    raise KeyError(f'Required modeling columns missing: {missing_required}')
season_column = 'Season' if 'Season' in data else 'season' if 'season' in data else None
if season_column is None:
    raise KeyError('A Season or season metadata column is required; seasons are not inferred silently.')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce', format='mixed').dt.normalize()
data['season'] = data[season_column].astype('string').str.strip().str.title().replace({'Autumn':'Fall'})
data['season'] = data['season'].fillna('Unknown')
for c in ['Plot','Strip']:
    data[c] = data[c].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
data['Treatment_days'] = data['Plot'].str.extract(r'^(14|28|42)(?:\.|$)', expand=False).fillna('Unknown')
for column in [TARGET] + NUMERIC_CANDIDATES:
    if column in data:
        data[column] = pd.to_numeric(data[column], errors='coerce').replace([np.inf,-np.inf], np.nan)
# The input is already quality-controlled. No new Huber or outcome-based exclusion is performed.
eligible = data[TARGET].notna() & data['Date'].notna() & data['Plot'].notna() & data['Strip'].notna()
save_table(data.loc[~eligible], '01_unusable_target_date_or_identifiers.csv')
analysis_data = data.loc[eligible].copy().reset_index(drop=True)
if analysis_data['Treatment_days'].eq('Unknown').any():
    raise ValueError('Some eligible plot IDs cannot be mapped to 14-, 28-, or 42-day treatments. Check Plot IDs.')
analysis_data['row_id'] = np.arange(len(analysis_data), dtype=int)
key_columns = ['Date','Plot','Strip'] + [c for c in ['Experiment','PrePost'] if c in analysis_data]
key = analysis_data[key_columns].astype(str).agg('|'.join, axis=1)
occurrence = key.groupby(key, sort=False).cumcount()
analysis_data['sample_id'] = (key + '|occ=' + occurrence.astype(str)).map(
    lambda v: hashlib.sha256(v.encode('utf-8')).hexdigest()[:20])
analysis_data['physical_strip_id'] = analysis_data['Plot'].astype(str)+'|'+analysis_data['Strip'].astype(str)
if not analysis_data['sample_id'].is_unique:
    raise AssertionError('Sample identifiers are not unique.')
assert not any(normalized(c) in excluded_names for c in analysis_data)
DATA_SHA256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
DATA_ORDER_SHA256 = hashlib.sha256('|'.join(analysis_data['sample_id']).encode()).hexdigest()
column_audit = pd.DataFrame({'column':analysis_data.columns,
    'dtype':[str(analysis_data[c].dtype) for c in analysis_data],
    'nonmissing_n':[int(analysis_data[c].notna().sum()) for c in analysis_data],
    'missing_n':[int(analysis_data[c].isna().sum()) for c in analysis_data],
    'unique_n':[int(analysis_data[c].nunique(dropna=True)) for c in analysis_data]})
predictor_audit = pd.DataFrame({'predictor':NUMERIC_CANDIDATES+CATEGORICAL_CANDIDATES})
predictor_audit['present_in_dataset'] = predictor_audit['predictor'].isin(analysis_data)
predictor_audit['predictor_class'] = predictor_audit['predictor'].map(FEATURE_GROUPS)
predictor_audit['display_label'] = predictor_audit['predictor'].map(FEATURE_LABELS)
predictor_audit['nonmissing_n'] = predictor_audit['predictor'].map(
    lambda c: int(analysis_data[c].notna().sum()) if c in analysis_data else 0)
save_table(column_audit,'01_dataset_column_audit.csv');save_table(predictor_audit,'01_predictor_space_audit.csv')
summary = pd.DataFrame({'item':['Input rows','Analysis rows','Excluded incomplete rows','Analysis columns',
                               'Sampling dates','Seasons','Plots','Strip-position labels','Physical plot-strip keys'],
    'value':[input_rows,len(analysis_data),input_rows-len(analysis_data),analysis_data.shape[1],
             analysis_data['Date'].nunique(),analysis_data['season'].nunique(),analysis_data['Plot'].nunique(),
             analysis_data['Strip'].nunique(),analysis_data['physical_strip_id'].nunique()]})
save_table(summary,'01_dataset_summary.csv')
save_table(analysis_data[['row_id','sample_id','Date','season','Treatment_days','Plot','Strip','physical_strip_id']],
           '01_sample_identity_manifest.csv')
# Number of independent date-level values is not the number of strip observations.
shared_rows=[]
for column in WEATHER_FEATURES+PHENO_NUMERIC_FEATURES+['season']:
    if column not in analysis_data: continue
    group_unique=analysis_data.groupby('Date')[column].nunique(dropna=False)
    shared_rows.append({'feature':column,'n_dates':len(group_unique),
                        'date_shared':bool(group_unique.max()<=1),'max_values_within_date':int(group_unique.max())})
DATE_SHARED_AUDIT=pd.DataFrame(shared_rows)
save_table(DATE_SHARED_AUDIT,'01_date_shared_predictor_audit.csv')
display(Markdown('## Dataset and predictor audit'))
display(summary);display(predictor_audit);display(DATE_SHARED_AUDIT)
print('Data SHA-256:',DATA_SHA256)
print('Date range:',analysis_data['Date'].min().date(),'to',analysis_data['Date'].max().date())
print('No previously rendered HTML results are reused. All subsequent values are recalculated.')


# ## Step 1.1 | Fix validation partitions
# Create the exact five-fold, leave-group-out, selected-date, and forward-date partitions once. Ridge, XGBoost, and ANN use the same evaluation rows in every protocol, isolating model-structure effects from split variability.

# Materialize splitting indices ONCE. All algorithms use these same positional arrays.
shared_kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
SHARED_KFOLD_SPLITS = register_splits(analysis_data,
    [(tr,te,f'Fold {i}') for i,(tr,te) in enumerate(shared_kfold.split(analysis_data),1)],'kfold_cv')
inner_generator = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
INNER_SPLITS = register_splits(analysis_data,
    [(tr,te,f'Inner fold {i}') for i,(tr,te) in enumerate(inner_generator.split(analysis_data),1)],
    'global_tuning',purpose='tuning')
INNER_CV_INDICES = [(tr,te) for tr,te,_ in INNER_SPLITS]
for protocol,column in [('leave_one_season_out','season'),
                        ('leave_one_treatment_out','Treatment_days'),
                        ('leave_one_strip_out',STRIP_GROUP_COLUMN)]:
    labels=analysis_data[column].astype(str).to_numpy()
    folds=[(tr,te,str(labels[te][0])) for tr,te in LeaveOneGroupOut().split(analysis_data,groups=labels)]
    register_splits(analysis_data,folds,protocol)
available_dates=sorted(analysis_data['Date'].dt.strftime('%Y-%m-%d').unique())
missing_dates=sorted(set(TARGETED_DATE_HOLDOUT_DATES)-set(available_dates))
if missing_dates:
    raise ValueError(f'Requested dates absent: {missing_dates}. Correct the configuration; no dates are substituted.')
selected=[]
for date in TARGETED_DATE_HOLDOUT_DATES:
    mask=analysis_data['Date'].eq(pd.Timestamp(date)).to_numpy()
    selected.append((np.flatnonzero(~mask),np.flatnonzero(mask),date))
register_splits(analysis_data,selected,'targeted_date_holdout')
latest_dates=available_dates[-FORWARD_HOLDOUT_N_DATES:]
forward=[]
for date in latest_dates:
    when=pd.Timestamp(date)
    train=np.flatnonzero(analysis_data['Date'].lt(when).to_numpy())
    test=np.flatnonzero(analysis_data['Date'].eq(when).to_numpy())
    forward.append((train,test,date))
register_splits(analysis_data,forward,'forward_date_holdout')
all_split_summaries=pd.concat(SPLIT_SUMMARIES.values(),ignore_index=True)
save_table(all_split_summaries,'00_all_split_counts_and_date_overlap.csv')
display(Markdown('## Exact training and testing sample counts'))
display(all_split_summaries)
print('All model families share the same outer folds and the same inner CV folds.')
print('Tuning uses the full dataset, as in the supplied source: subsequent comparisons are NON-NESTED.')
print('Forward fitting uses earlier dates, but full-data-selected hyperparameters prevent a fully prospective claim.')


# ## Step 2 | Descriptive analysis
# Summarize the response distribution and experimental coverage by season, treatment, plot, strip, and date. These summaries describe the modeled population before inferential screening or machine learning.

# ============================================================
# STEP 2. Descriptive data distribution and grouped summaries
# ============================================================

VI_WEATHER_SUMMARY_VARIABLES = [
    feature for feature in VI_FEATURES + WEATHER_FEATURES
    if feature in analysis_data.columns
]


def grouped_mean_median_table(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    variables: Sequence[str],
) -> pd.DataFrame:
    group_columns = list(group_columns)
    variables = [variable for variable in variables if variable in frame.columns]
    grouped = frame.groupby(group_columns, dropna=False, observed=False)
    summary = grouped[variables].agg(["mean", "median"])
    summary.columns = [f"{variable}_{statistic}" for variable, statistic in summary.columns]
    summary = summary.reset_index()
    counts = grouped.size().rename("n_records").reset_index()
    return counts.merge(summary, on=group_columns, how="left")


# Data-distribution table across the principal sampling dimensions
rows = []
for dimension, column in [
    ("Date", "Date"),
    ("Season", "season"),
    ("Treatment", "Treatment_days"),
    ("Plot", "Plot"),
    ("Strip", "Strip"),
]:
    for level, group in analysis_data.groupby(column, dropna=False, observed=False):
        rows.append({
            "dimension": dimension,
            "level": level.strftime("%Y-%m-%d") if isinstance(level, pd.Timestamp) else str(level),
            "n_records": int(len(group)),
            "percent_of_dataset": 100.0 * len(group) / len(analysis_data),
            "biomass_mean": float(group[TARGET].mean()),
            "biomass_median": float(group[TARGET].median()),
        })
data_distribution_table = pd.DataFrame(rows)
save_table(data_distribution_table, "02_data_distribution_by_sampling_dimension.csv")

# Mean and median VI/weather conditions by date, season, and treatment
summary_by_date = grouped_mean_median_table(analysis_data, ["Date"], VI_WEATHER_SUMMARY_VARIABLES)
summary_by_date["Date"] = pd.to_datetime(summary_by_date["Date"]).dt.strftime("%Y-%m-%d")
summary_by_season = grouped_mean_median_table(analysis_data, ["season"], VI_WEATHER_SUMMARY_VARIABLES)
summary_by_treatment = grouped_mean_median_table(analysis_data, ["Treatment_days"], VI_WEATHER_SUMMARY_VARIABLES)

save_table(summary_by_date, "02_vi_weather_mean_median_by_date.csv")
save_table(summary_by_season, "02_vi_weather_mean_median_by_season.csv")
save_table(summary_by_treatment, "02_vi_weather_mean_median_by_treatment.csv")

display(Markdown('## Data distribution by sampling dimension'))
display(data_distribution_table.round(3))
for title,table in [('Date',summary_by_date),('Season',summary_by_season),('Treatment',summary_by_treatment)]:
    display(Markdown(f'### Vegetation-index and weather summaries by {title.lower()}'))
    display(table.round(3))


# ## Step 3 | Exploratory analysis
# Visualize biomass distributions across season, plot, and strip to identify heterogeneity in the experimental response. These figures are descriptive and do not determine predictor inclusion.

# ============================================================
# STEP 3. Biomass distributions and Pearson correlation
# ============================================================

values = analysis_data[TARGET].dropna()
mean_value = float(values.mean())
median_value = float(values.median())
sd_value = float(values.std(ddof=1))
season_order = [season for season in ["Spring", "Summer", "Fall"] if season in analysis_data["season"].unique()]
treatment_order = [value for value in ["14", "28", "42"] if value in analysis_data["Treatment_days"].unique()]
plot_order = sorted(analysis_data["Plot"].unique(), key=lambda value: (int(str(value).split(".")[0]), float(value)))
strip_order = sorted(analysis_data["Strip"].unique(), key=lambda value: int(value))
strip_colors = distinct_colors(len(strip_order))

season_counts = analysis_data["season"].value_counts().to_dict()
treatment_counts = analysis_data["Treatment_days"].value_counts().to_dict()
summary_text = (f"Overall n = {len(values):,}\n"
                f"SD = {sd_value:,.0f} kg ha$^{{-1}}$\n"
                f"Range = {values.min():,.0f}-{values.max():,.0f} kg ha$^{{-1}}$\n"
                + "\n".join(f"{s}: n = {season_counts.get(s,0):,}" for s in season_order) + "\n"
                + "\n".join(f"{t}-day treatment: n = {treatment_counts.get(t,0):,}" for t in treatment_order))


# Overall biomass distribution with one integrated legend box
fig, ax = plt.subplots(figsize=(10.8, 7.2))
ax.hist(values, bins=28, color="#0072B2", alpha=0.88, edgecolor="white")
mean_line = ax.axvline(mean_value, color="#D55E00", linewidth=2.5)
median_line = ax.axvline(median_value, color="#009E73", linewidth=2.5, linestyle="--")
summary_handle = Patch(facecolor="none", edgecolor="none")
ax.legend(
    [mean_line, median_line, summary_handle],
    [f"Mean = {mean_value:,.0f}", f"Median = {median_value:,.0f}", summary_text],
    loc="upper right", frameon=True, framealpha=0.97, edgecolor="#BDBDBD",
    fontsize=13.2, handlelength=2.5, borderpad=1.0,
)
ax.set_xlabel(FEATURE_LABELS[TARGET])
ax.set_ylabel("Number of observations")
ax.tick_params(axis="both", labelsize=15)
fig.tight_layout()
save_figure(fig, "01_biomass_distribution")

# Biomass by season
fig, ax = plt.subplots(figsize=(9.8, 6.8))
draw_boxplot(
    data=analysis_data, x="season", y=TARGET, order=season_order,
    palette=[SEASON_COLORS[value] for value in season_order], ax=ax, width=0.62,
)
ax.set_xlabel("Season")
ax.set_ylabel(FEATURE_LABELS[TARGET])
ax.tick_params(axis="both", labelsize=15)
ax.legend(
    handles=[Patch(facecolor=SEASON_COLORS[value], label=value) for value in season_order],
    loc="upper right", frameon=True, edgecolor="#BDBDBD", fontsize=13,
)
fig.tight_layout()
save_figure(fig, "02_biomass_by_season")

# Biomass by treatment
fig, ax = plt.subplots(figsize=(9.8, 6.8))
draw_boxplot(
    data=analysis_data, x="Treatment_days", y=TARGET, order=treatment_order,
    palette=[TREATMENT_COLORS[value] for value in treatment_order], ax=ax, width=0.62,
)
ax.set_xlabel("Treatment (days)")
ax.set_ylabel(FEATURE_LABELS[TARGET])
ax.tick_params(axis="both", labelsize=15)
ax.legend(
    handles=[Patch(facecolor=TREATMENT_COLORS[value], label=f"{value} days") for value in treatment_order],
    title="Treatment", loc="upper left", frameon=True, edgecolor="#BDBDBD",
    fontsize=13, title_fontsize=14,
)
fig.tight_layout()
save_figure(fig, "03_biomass_by_Treatment")

# Biomass by plot, colored by treatment
plot_palette = [TREATMENT_COLORS.get(str(plot).split(".")[0], "#999999") for plot in plot_order]
fig, ax = plt.subplots(figsize=(15.5, 7.0))
draw_boxplot(data=analysis_data, x="Plot", y=TARGET, order=plot_order, palette=plot_palette, ax=ax, width=0.68)
ax.set_xlabel("Plot")
ax.set_ylabel(FEATURE_LABELS[TARGET])
ax.tick_params(axis="x", labelrotation=0, labelsize=13)
ax.tick_params(axis="y", labelrotation=0, labelsize=14)
ax.legend(
    handles=[Patch(facecolor=TREATMENT_COLORS[value], label=f"{value} days") for value in treatment_order],
    title="Treatment", loc="upper left", frameon=True, edgecolor="#BDBDBD",
    fontsize=13, title_fontsize=14,
)
fig.tight_layout()
save_figure(fig, "03_biomass_by_Plot")

# Biomass by strip
fig, ax = plt.subplots(figsize=(18.5, 7.0))
draw_boxplot(data=analysis_data, x="Strip", y=TARGET, order=strip_order, palette=strip_colors, ax=ax, width=0.68)
ax.set_xlabel("Strip")
ax.set_ylabel(FEATURE_LABELS[TARGET])
ax.tick_params(axis="x", labelrotation=0, labelsize=12.5)
ax.tick_params(axis="y", labelrotation=0, labelsize=14)
fig.tight_layout()
save_figure(fig, "03_biomass_by_Strip")

# Four-panel biomass distribution figure: overall, season, all 12 plots, strip
fig, axes = plt.subplots(2, 2, figsize=(26, 17))

# A. Overall biomass distribution
ax = axes[0, 0]
ax.hist(values, bins=28, color="#0072B2", alpha=0.88, edgecolor="white")
mean_panel_line = ax.axvline(mean_value, color="#D55E00", linewidth=2.7)
median_panel_line = ax.axvline(median_value, color="#009E73", linewidth=2.7, linestyle="--")
ax.legend(
    [mean_panel_line, median_panel_line, summary_handle],
    [f"Mean = {mean_value:,.0f}", f"Median = {median_value:,.0f}", summary_text],
    loc="upper right",
    frameon=True,
    framealpha=0.97,
    edgecolor="#BDBDBD",
    fontsize=19,
    handlelength=2.6,
    borderpad=1.0,
)
ax.set_xlabel(FEATURE_LABELS[TARGET], fontsize=25)
ax.set_ylabel("Number of observations", fontsize=25)
ax.tick_params(axis="both", labelsize=21)
ax.set_title("A", fontsize=30, fontweight="bold", pad=18, loc="center")

# B. Biomass by season
ax = axes[0, 1]
draw_boxplot(
    data=analysis_data,
    x="season",
    y=TARGET,
    order=season_order,
    palette=[SEASON_COLORS[value] for value in season_order],
    ax=ax,
    width=0.62,
)
ax.set_xlabel("Season", fontsize=25)
ax.set_ylabel(FEATURE_LABELS[TARGET], fontsize=25)
ax.tick_params(axis="both", labelsize=21)
ax.legend(
    handles=[Patch(facecolor=SEASON_COLORS[value], label=value) for value in season_order],
    loc="upper right",
    frameon=True,
    edgecolor="#BDBDBD",
    fontsize=20,
)
ax.set_title("B", fontsize=30, fontweight="bold", pad=18, loc="center")

# C. Biomass across all 12 plots, colored by treatment
ax = axes[1, 0]
draw_boxplot(
    data=analysis_data,
    x="Plot",
    y=TARGET,
    order=plot_order,
    palette=plot_palette,
    ax=ax,
    width=0.68,
)
ax.set_xlabel("Plot", fontsize=25)
ax.set_ylabel(FEATURE_LABELS[TARGET], fontsize=25)
ax.tick_params(axis="x", labelrotation=0, labelsize=20)
ax.tick_params(axis="y", labelrotation=0, labelsize=21)
ax.legend(
    handles=[Patch(facecolor=TREATMENT_COLORS[value], label=f"{value} days") for value in treatment_order],
    title="Treatment",
    loc="upper left",
    frameon=True,
    edgecolor="#BDBDBD",
    fontsize=19,
    title_fontsize=25,
)
ax.set_title("C", fontsize=30, fontweight="bold", pad=18, loc="center")

# D. Biomass by strip
ax = axes[1, 1]
draw_boxplot(
    data=analysis_data,
    x="Strip",
    y=TARGET,
    order=strip_order,
    palette=strip_colors,
    ax=ax,
    width=0.68,
)
ax.set_xlabel("Strip", fontsize=25)
ax.set_ylabel(FEATURE_LABELS[TARGET], fontsize=25)
ax.tick_params(axis="x", labelrotation=0, labelsize=19)
ax.tick_params(axis="y", labelrotation=0, labelsize=21)
ax.set_title("D", fontsize=30, fontweight="bold", pad=18, loc="center")

fig.subplots_adjust(wspace=0.26, hspace=0.34, top=0.965, bottom=0.075, left=0.075, right=0.99)
save_figure(fig, "03z_exploratory_summary_season_plot_strip_2x2_panel")


# ## Step 3A | Pearson correlation structure
# Quantify pairwise linear association among biomass and numeric candidate predictors using Pearson correlation. The heatmap is descriptive; formal redundancy control is performed in Step 5.

# ============================================================
# STEP 3A. Pearson correlation heatmap - previous manuscript design
# ============================================================

correlation_features = [TARGET] + [
    feature
    for feature in HEIGHT_FEATURES + VI_FEATURES + WEATHER_FEATURES + PHENO_NUMERIC_FEATURES
    if feature in analysis_data.columns
]
correlation_features = list(dict.fromkeys(correlation_features))

# Defensive removal of variables excluded from the full analysis.
correlation_features = [
    feature for feature in correlation_features
    if feature not in {'Dry Matter %', 'SRre_mean'}
]

correlation_data = analysis_data[correlation_features].apply(pd.to_numeric, errors='coerce')
correlation_matrix = correlation_data.corr(method='pearson')

# Use the same long scientific labels used in the previous manuscript heatmap.
heatmap_labels = [FEATURE_LABELS.get(feature, str(feature)) for feature in correlation_features]
heatmap_classes = [FEATURE_GROUPS[feature] for feature in correlation_features]
n = len(correlation_features)

# Previous negative-red / positive-blue palette.
cmap = sns.diverging_palette(
    15,   # red negative end
    220,  # blue positive end
    s=95,
    l=48,
    as_cmap=True,
)

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

fig_width = max(19.0, 0.82 * n)
fig_height = max(17.0, 0.82 * n)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

hm = sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.75,
    linecolor='white',
    cbar=True,
    cbar_kws={
        'shrink': 0.46,
        'pad': 0.030,
        'aspect': 16,
        'label': 'Pearson r',
    },
    annot=False,
    ax=ax,
)

# Larger in-cell coefficients, as requested.
for row in range(n):
    for col in range(row + 1):
        value = correlation_matrix.iloc[row, col]
        if not np.isfinite(value):
            continue
        if abs(value) < 0.005:
            value = 0.0
        ax.text(
            col + 0.5,
            row + 0.5,
            f'{value:.2f}',
            ha='center', va='center',
            fontsize=14.5,
            color='white' if abs(value) >= 0.60 else '#222222',
            fontweight='normal',
        )

ax.set_xticklabels(
    heatmap_labels,
    rotation=90,
    ha='center',
    fontsize=18,
    fontweight='normal',
)
ax.set_yticklabels(
    heatmap_labels,
    rotation=0,
    fontsize=18,
    fontweight='normal',
)

# Increase spacing between y-axis labels and the class-color strip.
ax.tick_params(axis='x', length=0, pad=10)
ax.tick_params(axis='y', length=0, pad=18)

# Variable-class strips reproduce the previous figure.
for idx, class_name in enumerate(heatmap_classes):
    color = FEATURE_CLASS_COLORS[class_name]
    ax.add_patch(
        patches.Rectangle(
            (idx, -0.44), 1.0, 0.44,
            facecolor=color, edgecolor='none',
            clip_on=False,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (-0.44, idx), 0.44, 1.0,
            facecolor=color, edgecolor='none',
            clip_on=False,
        )
    )

class_order = list(dict.fromkeys(heatmap_classes))
legend_handles = [
    Patch(facecolor=FEATURE_CLASS_COLORS[class_name], edgecolor='none', label=class_name)
    for class_name in class_order
]
ax.legend(
    handles=legend_handles,
    title='Variable class',
    loc='upper right',
    bbox_to_anchor=(0.985, 0.985),
    frameon=True,
    framealpha=0.96,
    facecolor='white',
    edgecolor='#BDBDBD',
    fontsize=17,
    title_fontsize=19,
    borderpad=0.65,
    labelspacing=0.45,
)

cbar = hm.collections[0].colorbar
cbar.set_ticks([-1, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1])
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Pearson r', fontsize=18, labelpad=10)

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')

fig.subplots_adjust(
    left=0.25,
    right=0.94,
    bottom=0.27,
    top=0.965,
)

save_figure(fig, '04_final_enlarged_pearson_correlation_heatmap')

corr_export = correlation_matrix.copy()
corr_export.index = [FEATURE_LABELS.get(feature, feature) for feature in corr_export.index]
corr_export.columns = [FEATURE_LABELS.get(feature, feature) for feature in corr_export.columns]
save_table(corr_export.rename_axis('Variable').reset_index(), '04_pearson_correlation_matrix.csv')


# ## Step 4 | Source baselines
# Fit two unscreened linear benchmarks on identical five-fold splits: canopy height only, and canopy height + PlanetScope VIs. These baselines quantify the predictive value available before multisource screening and nonlinear model development.

# ============================================================
# STEP 4. Five-fold conventional linear-regression baselines
# ============================================================

baseline_splits = SHARED_KFOLD_SPLITS
baseline_metric_rows = []
baseline_prediction_frames = []
baseline_selection_rows = []

for fold_number, (train_index, test_index, _) in enumerate(baseline_splits, start=1):
    train_data = analysis_data.iloc[train_index].copy()
    test_data = analysis_data.iloc[test_index].copy()

    # Height-only linear regression
    height_median = float(train_data["PT_Height(mm)"].median())
    X_train_height = train_data[["PT_Height(mm)"]].fillna(height_median)
    X_test_height = test_data[["PT_Height(mm)"]].fillna(height_median)
    height_model = LinearRegression().fit(X_train_height, train_data[TARGET])
    height_predictions = height_model.predict(X_test_height)
    baseline_metric_rows.append(metrics_row(
        test_data[TARGET], height_predictions, BASELINE_HEIGHT_NAME, "baseline_fivefold_cv",
        fold_number, f"Fold {fold_number}", ["PT_Height(mm)"],
    ))
    baseline_prediction_frames.append(pd.DataFrame({
        "row_id": test_data["row_id"].to_numpy(),
        "observed": test_data[TARGET].to_numpy(dtype=float),
        "predicted": height_predictions,
        "model": BASELINE_HEIGHT_NAME,
        "protocol": "baseline_fivefold_cv",
        "fold": str(fold_number),
        "held_out_group": f"Fold {fold_number}",
    }))

    # Height + PlanetScope VI linear stepwise regression
    stepwise_fit = stepwise_aic_fit(
        train_data, candidate_features=HEIGHT_FEATURES + VI_FEATURES, forced_features=HEIGHT_FEATURES,
    )
    stepwise_predictions = stepwise_predict(stepwise_fit, test_data)
    baseline_metric_rows.append(metrics_row(
        test_data[TARGET], stepwise_predictions, BASELINE_STEPWISE_NAME, "baseline_fivefold_cv",
        fold_number, f"Fold {fold_number}", stepwise_fit["selected_features"],
    ))
    baseline_prediction_frames.append(pd.DataFrame({
        "row_id": test_data["row_id"].to_numpy(),
        "observed": test_data[TARGET].to_numpy(dtype=float),
        "predicted": stepwise_predictions,
        "model": BASELINE_STEPWISE_NAME,
        "protocol": "baseline_fivefold_cv",
        "fold": str(fold_number),
        "held_out_group": f"Fold {fold_number}",
    }))
    baseline_selection_rows.append({
        "fold": fold_number,
        "selected_features": " | ".join(stepwise_fit["selected_features"]),
        "AIC": stepwise_fit["aic"],
    })

baseline_metrics = pd.DataFrame(baseline_metric_rows)
baseline_metrics_export = append_average_accuracy_rows(baseline_metrics)
baseline_predictions = pd.concat(baseline_prediction_frames, ignore_index=True)
baseline_selection = pd.DataFrame(baseline_selection_rows)
baseline_average = average_metrics_table(baseline_metrics)

save_table(baseline_metrics_export, "baseline_01_fivefold_metrics_with_average_accuracy.csv")
save_table(baseline_average, "baseline_02_average_fivefold_metrics.csv")
save_table(baseline_predictions, "baseline_03_fivefold_predictions.csv")
save_table(baseline_selection, "baseline_04_stepwise_selected_features_by_fold.csv")

display(Markdown("## Five-fold baseline metrics with average-accuracy rows"))
display(baseline_metrics_export.round(3))
display(Markdown("## Average baseline performance with fold variation"))
display(baseline_average.round(3))

baseline_models = [BASELINE_HEIGHT_NAME, BASELINE_STEPWISE_NAME]
for model_name in baseline_models:
    model_data = baseline_predictions[baseline_predictions["model"] == model_name]
    for fold in sorted(model_data["fold"].unique(), key=int):
        subset = model_data[model_data["fold"] == fold]
        plot_single_repetition_scatter(
            subset, f"{model_name}: Fold {fold}", BASELINE_COLORS[model_name],
            f"baseline_{safe_name(model_name)}_fold_{fold}", observation_label=f"Fold {fold}",
        )

# Publication 2 x 5 baseline panel
plot_collective_protocol_matrix(
    baseline_predictions, "baseline_fivefold_cv", "baseline_01_fivefold_prediction_panels",
    model_order=baseline_models,
)

# One panel per baseline containing every fold-specific fitted line
plot_fold_specific_lines_panel(
    baseline_predictions, "baseline_fivefold_cv", "baseline_02_fold_specific_fitted_lines_panel",
    model_order=baseline_models,
)

baseline_summary = plot_average_stability(
    baseline_metrics, "baseline_fivefold_cv", "baseline_03_average_fivefold_metric_stability",
    model_order=baseline_models,
)
save_table(baseline_summary, "baseline_05_average_metric_stability_data.csv")


# ## Step 5 | Predictor screening and selection
# Screen numeric predictors using complementary criteria. Highly redundant pairs with |Pearson r| ≥ 0.90 are pruned by retaining the member more strongly associated with biomass; VIF is then iteratively reduced toward ≤ 10 to control multicollinearity. Mutual information ranks possible nonlinear dependence with biomass and is used as a complementary safeguard, not as a significance test. Categorical season is retained separately when more than one level is present. The same retained predictor logic is used for all model families.

# ============================================================
# STEP 5. Full-data feature screening for configuration and audit
# ============================================================

global_screening = screen_features(analysis_data, display_outputs=False, tag='complete-data configuration screen')
GLOBAL_SELECTED_FEATURES = global_screening["selected_features"]

for key, filename in [
    ("feature_catalog", "05_feature_catalog.csv"),
    ("raw_vif", "05_raw_vif.csv"),
    ("mutual_information", "05_mutual_information.csv"),
    ("correlation_removed", "05_pearson_removed_features.csv"),
    ("high_correlation_pairs", "05_high_correlation_pairs.csv"),
    ("vif_steps", "05_vif_pruning_steps.csv"),
    ("final_vif", "05_retained_predictor_vif.csv"),
]:
    save_table(global_screening[key], filename)

print(f"Candidate numeric predictors: {len(NUMERIC_CANDIDATES)}")
print(f"Final retained predictors: {len(GLOBAL_SELECTED_FEATURES)}")
print("Selected features:", GLOBAL_SELECTED_FEATURES)

raw_vif = global_screening["raw_vif"].copy()
mutual_information = global_screening["mutual_information"].copy()
final_vif = global_screening["final_vif"].copy()
feature_catalog = global_screening["feature_catalog"].copy()

# Compact manuscript-facing summary; complete audit tables are exported above.
selected_summary = feature_catalog.loc[feature_catalog["selected_for_model"],
    ["feature_label", "abs_pearson_with_biomass", "mutual_information", "final_VIF"]].copy()
selected_summary.columns = ["Retained predictor", "|Pearson r| with biomass", "Mutual information", "Final VIF"]
display(Markdown("### Retained predictor set"))
display(selected_summary.round(3))

plot_raw = raw_vif.sort_values("raw_VIF", ascending=True).copy()
plot_raw["plot_VIF"] = plot_raw["raw_VIF"].replace(np.inf, 2_500_000).clip(upper=2_500_000)
plot_raw["bar_color"] = np.where(plot_raw["raw_VIF"] > VIF_THRESHOLD, "#C63D4F", "#4A86BE")
plot_mi = mutual_information.merge(
    feature_catalog[["feature", "selected_for_model"]], on="feature", how="left"
).sort_values("mutual_information", ascending=True)
plot_mi["status"] = np.where(plot_mi["selected_for_model"].fillna(False), "retained", "screened")
plot_mi["bar_color"] = np.where(plot_mi["status"] == "retained", "#4A86BE", "#BDBDBD")
plot_final = final_vif.sort_values("VIF", ascending=True).copy()

# Individual raw VIF
fig, ax = plt.subplots(figsize=(11.5, 9.8))
ax.barh([FEATURE_LABELS.get(value, str(value)) for value in plot_raw["feature"]], plot_raw["plot_VIF"],
        color=plot_raw["bar_color"], height=0.72)
ax.set_xscale("log")
ax.axvline(VIF_THRESHOLD, color="black", linestyle=":", linewidth=2.0, label=f"VIF threshold = {VIF_THRESHOLD:g}")
for index, row in plot_raw.reset_index(drop=True).iterrows():
    label = 'inf' if not np.isfinite(row['raw_VIF']) else f"{row['raw_VIF']:,.1f}"
    ax.text(row["plot_VIF"] * 1.05, index, label, va="center", ha="left", fontsize=16)
ax.set_xlabel("Variance inflation factor (log scale)")
ax.set_ylabel("Predictor")
ax.tick_params(axis="y", labelrotation=0, labelsize=13)
ax.legend(loc="lower right", frameon=True, edgecolor="#BDBDBD")
fig.tight_layout()
save_figure(fig, "06a_raw_vif_all_predictors")

# Individual mutual information
fig, ax = plt.subplots(figsize=(11.5, 9.8))
ax.barh([FEATURE_LABELS.get(value, str(value)) for value in plot_mi["feature"]], plot_mi["mutual_information"],
        color=plot_mi["bar_color"], height=0.72)
for index, row in plot_mi.reset_index(drop=True).iterrows():
    ax.text(row["mutual_information"] + 0.01, index, row["status"], va="center", ha="left", fontsize=16)
ax.set_xlabel("Mutual information score")
ax.set_ylabel("Predictor")
ax.tick_params(axis="y", labelrotation=0, labelsize=13)
fig.tight_layout()
save_figure(fig, "06b_mutual_information")

# Individual retained VIF
fig, ax = plt.subplots(figsize=(11.5, 8.2))
ax.barh([FEATURE_LABELS.get(value, str(value)) for value in plot_final["feature"]], plot_final["VIF"],
        color="#4A86BE", height=0.72)
ax.set_xscale("log")
ax.axvline(VIF_THRESHOLD, color="black", linestyle=":", linewidth=2.0, label=f"VIF threshold = {VIF_THRESHOLD:g}")
for index, value in enumerate(plot_final["VIF"]):
    ax.text(value * 1.05, index, f"{value:.1f}", va="center", ha="left", fontsize=16)
ax.set_xlabel("Variance inflation factor (log scale)")
ax.set_ylabel("Predictor")
ax.tick_params(axis="y", labelrotation=0, labelsize=13)
ax.legend(loc="lower right", frameon=True, edgecolor="#BDBDBD")
fig.tight_layout()
save_figure(fig, "06c_retained_predictor_vif")

# Three-panel feature-selection figure using the previous visual concept
fig, axes = plt.subplots(1, 3, figsize=(32, 12))
ax = axes[0]
ax.barh([FEATURE_LABELS.get(value, str(value)) for value in plot_raw["feature"]], plot_raw["plot_VIF"],
        color=plot_raw["bar_color"], height=0.72)
ax.set_xscale("log")
ax.axvline(VIF_THRESHOLD, color="black", linestyle=":", linewidth=2.0, label=f"VIF threshold = {VIF_THRESHOLD:g}")
for index, row in plot_raw.reset_index(drop=True).iterrows():
    label = 'inf' if not np.isfinite(row['raw_VIF']) else f"{row['raw_VIF']:,.1f}"
    ax.text(row["plot_VIF"] * 1.05, index, label, va="center", ha="left", fontsize=22)
ax.set_xlabel("Variance inflation factor (log scale)", fontsize=22)
ax.set_ylabel("Predictor", fontsize=22)
ax.tick_params(axis="y", labelrotation=0, labelsize=18)
ax.legend(loc="lower right", frameon=True, edgecolor="#BDBDBD", fontsize=16)
ax.set_title("A", fontsize=30, fontweight="bold", pad=18, loc="center")

ax = axes[1]
ax.barh([FEATURE_LABELS.get(value, str(value)) for value in plot_mi["feature"]], plot_mi["mutual_information"],
        color=plot_mi["bar_color"], height=0.72)
for index, row in plot_mi.reset_index(drop=True).iterrows():
    ax.text(row["mutual_information"] + 0.01, index, row["status"], va="center", ha="left", fontsize=22)
ax.set_xlabel("Mutual information score", fontsize=22)
ax.set_ylabel("Predictor", fontsize=22)
ax.tick_params(axis="y", labelrotation=0, labelsize=18)
ax.set_title("B", fontsize=30, fontweight="bold", pad=18, loc="center")

ax = axes[2]
ax.barh([FEATURE_LABELS.get(value, str(value)) for value in plot_final["feature"]], plot_final["VIF"],
        color="#4A86BE", height=0.72)
ax.set_xscale("log")
ax.axvline(VIF_THRESHOLD, color="black", linestyle=":", linewidth=2.0, label=f"VIF threshold = {VIF_THRESHOLD:g}")
for index, value in enumerate(plot_final["VIF"]):
    ax.text(value * 1.05, index, f"{value:.1f}", va="center", ha="left", fontsize=22)
ax.set_xlabel("Variance inflation factor (log scale)", fontsize=22)
ax.set_ylabel("Predictor", fontsize=22)
ax.tick_params(axis="y", labelrotation=0, labelsize=18)
ax.legend(loc="lower right", frameon=True, edgecolor="#BDBDBD", fontsize=16)
ax.set_title("C", fontsize=30, fontweight="bold", pad=18, loc="center")

fig.subplots_adjust(wspace=0.72, top=0.97, bottom=0.11, left=0.10, right=0.99)
save_figure(fig, "06d_feature_screening_three_panel")

violations=global_screening['feature_catalog'].loc[global_screening['feature_catalog']['final_VIF_above_threshold']]
if not violations.empty:
    print('Minimum-feature safeguard retained predictors above the VIF target:')
    display(violations[['feature','final_VIF']])


# ## Step 6 | Hyperparameter optimization
# Tune model complexity using the same five inner folds and RMSE as the optimization objective. Ridge uses a logarithmic alpha grid; XGBoost uses a broad randomized search over tree complexity, learning rate, sampling, and regularization; ANN uses randomized architecture and regularization search with early stopping. Hyperparameters are selected for predictive accuracy only, not to force any feature-importance ranking, and the selected settings are then held fixed across validation protocols.

# ============================================================
# STEP 6. Cross-validation hyperparameter optimization
# ============================================================

inner_cv = INNER_CV_INDICES
X_tune = analysis_data[GLOBAL_SELECTED_FEATURES]
y_tune = analysis_data[TARGET]

# Ridge grid search
ridge_pipeline = build_model_pipeline("Ridge", GLOBAL_SELECTED_FEATURES, {"alpha": 1.0})
ridge_search = GridSearchCV(
    ridge_pipeline,
    param_grid={"model__alpha": RIDGE_ALPHA_GRID},
    scoring="neg_root_mean_squared_error",
    cv=inner_cv,
    n_jobs=SEARCH_N_JOBS,
    refit=True,
)
ridge_search.fit(X_tune, y_tune)
ridge_results = pd.DataFrame(ridge_search.cv_results_)
ridge_best = {"alpha": float(ridge_search.best_params_["model__alpha"])}

# XGBoost randomized search
xgb_pipeline = build_model_pipeline("XGBoost", GLOBAL_SELECTED_FEATURES)
xgb_distributions = {
    # Ensemble size and shrinkage are searched jointly because smaller learning
    # rates generally require more trees.
    "model__n_estimators": randint(200, 1401),
    "model__max_depth": randint(2, 8),
    "model__learning_rate": loguniform(0.004, 0.20),
    # Stochastic row/feature sampling controls variance and overfitting.
    "model__subsample": uniform(0.60, 0.40),
    "model__colsample_bytree": uniform(0.60, 0.40),
    # Split constraints and explicit L1/L2 regularization control tree complexity.
    "model__min_child_weight": randint(1, 16),
    "model__gamma": uniform(0.0, 0.75),
    "model__reg_alpha": loguniform(1e-6, 10.0),
    "model__reg_lambda": loguniform(0.05, 80.0),
}
xgb_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=xgb_distributions,
    n_iter=XGB_N_ITER,
    scoring="neg_root_mean_squared_error",
    cv=inner_cv,
    random_state=RANDOM_STATE,
    n_jobs=SEARCH_N_JOBS,
    refit=True,
)
xgb_search.fit(X_tune, y_tune)
xgb_results = pd.DataFrame(xgb_search.cv_results_)
xgb_best = {key.replace("model__", ""): value for key, value in xgb_search.best_params_.items()}

# ANN randomized search
ann_pipeline = build_model_pipeline("ANN", GLOBAL_SELECTED_FEATURES)
ann_distributions = {
    "model__regressor__hidden_layer_sizes": [
        (32,), (64,), (128,),
        (32, 16), (64, 32), (128, 64),
        (64, 32, 16), (128, 64, 32),
    ],
    "model__regressor__activation": ["relu", "tanh"],
    "model__regressor__alpha": loguniform(1e-6, 1e-1),
    "model__regressor__learning_rate_init": loguniform(1e-4, 1e-2),
    "model__regressor__batch_size": [16, 32, 64, 128],
}
ann_search = RandomizedSearchCV(
    ann_pipeline,
    param_distributions=ann_distributions,
    n_iter=ANN_N_ITER,
    scoring="neg_root_mean_squared_error",
    cv=inner_cv,
    random_state=RANDOM_STATE,
    n_jobs=SEARCH_N_JOBS,
    refit=True,
)
ann_search.fit(X_tune, y_tune)
ann_results = pd.DataFrame(ann_search.cv_results_)
ann_best = {key.replace("model__regressor__", ""): value for key, value in ann_search.best_params_.items()}

BEST_PARAMETERS = {
    "Ridge": ridge_best,
    "XGBoost": xgb_best,
    "ANN": ann_best,
}

parameter_rows=[]
for model,parameters in BEST_PARAMETERS.items():
    for name,value in parameters.items():
        parameter_rows.append({'model':model,'parameter':name,'selected_value':decimal_text(value),
                               'active_for_solver':True,
                               'parameter_hash':parameter_signature(parameters)})
parameter_table=pd.DataFrame(parameter_rows)
save_table(parameter_table,'07_best_hyperparameters.csv')
save_table(ridge_results,'07_ridge_search_results.csv')
save_table(xgb_results,'07_xgboost_search_results.csv')
save_table(ann_results,'07_ann_search_results.csv')
with open(MODEL_DIR/'best_hyperparameters_v17_final.json','w',encoding='utf-8') as file:
    json.dump(plain_value(BEST_PARAMETERS),file,indent=2)
display(Markdown('### Optimized hyperparameters'))
display(parameter_table[['model','parameter','selected_value','active_for_solver']])
fit_allocation=[]
for fold,(tr,va,_) in enumerate(INNER_SPLITS,1):
    for model in MODEL_ORDER:
        n_validation=int(np.ceil(.15*len(tr))) if model=='ANN' else 0
        fit_allocation.append({'model':model,'inner_fold':fold,'fit_partition_n':len(tr),
            'scoring_partition_n':len(va),'internal_early_stopping_n':n_validation,
            'optimizer_training_n':len(tr)-n_validation,
            'internal_exact_ids':'not exposed by MLPRegressor' if model=='ANN' else 'not applicable'})
save_table(pd.DataFrame(fit_allocation),'07_model_tuning_sample_allocation.csv')
print(f'Tuning allocation exported for {len(fit_allocation):,} model-fold combinations.')

# Individual optimization figures and combined panel
ridge_plot = ridge_results.sort_values("param_model__alpha")
xgb_plot = xgb_results.sort_values("rank_test_score")
ann_plot = ann_results.sort_values("rank_test_score")
ridge_best_rmse = float(-ridge_search.best_score_)
xgb_best_rmse = float(-xgb_search.best_score_)
ann_best_rmse = float(-ann_search.best_score_)


def add_best_box(ax: plt.Axes, lines: Sequence[str]) -> None:
    ax.text(
        0.02, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
        fontsize=18, bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#BDBDBD", alpha=0.95),
    )

fig, ax = plt.subplots(figsize=(9.4, 6.8))
ax.plot(ridge_plot["param_model__alpha"].astype(float), -ridge_plot["mean_test_score"], marker="o",
        color=MODEL_COLORS["Ridge"], linewidth=2.4)
ax.axvline(ridge_best["alpha"], color="#B2182B", linestyle="--", linewidth=2.1)
ax.scatter([ridge_best["alpha"]], [ridge_best_rmse], marker="*", s=180, color="#B2182B", zorder=6)
ax.set_xscale("log")
ax.set_xlabel("Ridge alpha")
ax.set_ylabel("Cross-validated RMSE (kg ha$^{-1}$)")
ax.set_title("Ridge hyperparameter optimization")
add_best_box(ax, [f"Best alpha = {decimal_text(ridge_best['alpha'])}", f"Best RMSE = {ridge_best_rmse:,.2f}"])
fig.tight_layout()
save_figure(fig, "07a_ridge_hyperparameter_optimization")

fig, ax = plt.subplots(figsize=(9.4, 6.8))
ax.plot(np.arange(1, len(xgb_plot) + 1), -xgb_plot["mean_test_score"], marker="o",
        color=MODEL_COLORS["XGBoost"], linewidth=2.4)
ax.scatter([1], [xgb_best_rmse], marker="*", s=180, color="#B2182B", zorder=6)
ax.set_xlabel("Ranked XGBoost configuration")
ax.set_ylabel("Cross-validated RMSE (kg ha$^{-1}$)")
ax.set_title("XGBoost hyperparameter optimization")
add_best_box(ax, [f"Best RMSE = {xgb_best_rmse:,.2f}", f"Depth = {xgb_best.get('max_depth')}",
                  f"Learning rate = {float(xgb_best.get('learning_rate', np.nan)):.4f}",
                  f"Trees = {xgb_best.get('n_estimators')}"])
fig.tight_layout()
save_figure(fig, "07b_xgboost_hyperparameter_optimization")

fig, ax = plt.subplots(figsize=(9.4, 6.8))
ax.plot(np.arange(1, len(ann_plot) + 1), -ann_plot["mean_test_score"], marker="o",
        color=MODEL_COLORS["ANN"], linewidth=2.4)
ax.scatter([1], [ann_best_rmse], marker="*", s=180, color="#B2182B", zorder=6)
ax.set_xlabel("Ranked ANN configuration")
ax.set_ylabel("Cross-validated RMSE (kg ha$^{-1}$)")
ax.set_title("ANN hyperparameter optimization")
add_best_box(ax, [f"Best RMSE = {ann_best_rmse:,.2f}", f"Layers = {ann_best.get('hidden_layer_sizes')}",
                  f"Alpha = {float(ann_best.get('alpha', np.nan)):.8f}",
                  f"Learning rate = {float(ann_best.get('learning_rate_init', np.nan)):.6f}"])
fig.tight_layout()
save_figure(fig, "07c_ann_hyperparameter_optimization")

fig, axes = plt.subplots(1, 3, figsize=(27, 8.2))
axes[0].plot(ridge_plot["param_model__alpha"].astype(float), -ridge_plot["mean_test_score"], marker="o",
             color=MODEL_COLORS["Ridge"], linewidth=2.4)
axes[0].axvline(ridge_best["alpha"], color="#B2182B", linestyle="--", linewidth=2.1)
axes[0].scatter([ridge_best["alpha"]], [ridge_best_rmse], marker="*", s=180, color="#B2182B", zorder=6)
axes[0].set_xscale("log")
axes[0].set_xlabel("Ridge alpha")
axes[0].set_ylabel("Cross-validated RMSE (kg ha$^{-1}$)")
axes[0].set_title("Ridge hyperparameter optimization")
add_best_box(axes[0], [f"Best alpha = {decimal_text(ridge_best['alpha'])}", f"Best RMSE = {ridge_best_rmse:,.2f}"])

axes[1].plot(np.arange(1, len(xgb_plot) + 1), -xgb_plot["mean_test_score"], marker="o",
             color=MODEL_COLORS["XGBoost"], linewidth=2.4)
axes[1].scatter([1], [xgb_best_rmse], marker="*", s=180, color="#B2182B", zorder=6)
axes[1].set_xlabel("Ranked XGBoost configuration")
axes[1].set_ylabel("Cross-validated RMSE (kg ha$^{-1}$)")
axes[1].set_title("XGBoost hyperparameter optimization")
add_best_box(axes[1], [f"Best RMSE = {xgb_best_rmse:,.2f}", f"Depth = {xgb_best.get('max_depth')}",
                       f"Learning rate = {float(xgb_best.get('learning_rate', np.nan)):.6f}",
                       f"Trees = {xgb_best.get('n_estimators')}"])

axes[2].plot(np.arange(1, len(ann_plot) + 1), -ann_plot["mean_test_score"], marker="o",
             color=MODEL_COLORS["ANN"], linewidth=2.4)
axes[2].scatter([1], [ann_best_rmse], marker="*", s=180, color="#B2182B", zorder=6)
axes[2].set_xlabel("Ranked ANN configuration")
axes[2].set_ylabel("Cross-validated RMSE (kg ha$^{-1}$)")
axes[2].set_title("ANN hyperparameter optimization")
add_best_box(axes[2], [f"Best RMSE = {ann_best_rmse:,.2f}", f"Layers = {ann_best.get('hidden_layer_sizes')}",
                       f"Alpha = {float(ann_best.get('alpha', np.nan)):.8f}",
                       f"Learning rate = {float(ann_best.get('learning_rate_init', np.nan)):.6f}"])
for ax in axes:
    ax.tick_params(axis="both", labelsize=19)
    ax.tick_params(axis="y", labelrotation=0)
fig.subplots_adjust(wspace=0.30, top=0.91, bottom=0.16, left=0.07, right=0.99)
save_figure(fig, "07z_hyperparameter_optimization_three_panel")


# ## Step 7 | Five-fold out-of-fold evaluation
# Refit each model within each training fold and predict only the corresponding held-out fold. Pooling these out-of-fold predictions provides a common assessment of model performance and fold-to-fold stability.

# ============================================================
# STEP 7. Five-fold model validation
# ============================================================

kfold_splits = SHARED_KFOLD_SPLITS

kfold_metrics, kfold_predictions, KFOLD_MODELS = run_validation_protocol(
    analysis_data, kfold_splits, protocol="kfold_cv", best_parameters=BEST_PARAMETERS, store_models=True,
)
kfold_average = average_metrics_table(kfold_metrics)
kfold_metrics_export = append_average_accuracy_rows(kfold_metrics)

save_table(kfold_metrics_export, "08_kfold_metrics_with_average_accuracy.csv")
save_table(kfold_average, "08_kfold_average_metrics.csv")
save_table(kfold_predictions, "08_kfold_predictions.csv")

display(Markdown("## Five-fold metrics with average-accuracy rows"))
display(kfold_metrics_export.round(3))
display(Markdown("## Average five-fold performance"))
display(kfold_average.round(3))

for model_name in MODEL_ORDER:
    model_data = kfold_predictions[kfold_predictions["model"] == model_name]
    for group in list(dict.fromkeys(model_data["held_out_group"].tolist())):
        subset = model_data[model_data["held_out_group"] == group]
        plot_single_repetition_scatter(
            subset, f"{model_display_name(model_name)}: {group}", MODEL_COLORS[model_name],
            f"kfold_{safe_name(model_name)}_{safe_name(group)}", observation_label=group,
        )
    plot_repetition_grid(
        kfold_predictions, model_name, "kfold_cv", f"kfold_cv_{safe_name(model_name)}_five_fold_panel",
        ncols_override=5,
    )

plot_collective_protocol_matrix(
    kfold_predictions, "kfold_cv", "kfold_cv_all_models_3x5_panel", model_order=MODEL_ORDER,
)
plot_fold_specific_lines_panel(
    kfold_predictions, "kfold_cv", "kfold_cv_fold_specific_fitted_lines_three_model_panel",
)

kfold_stability = plot_average_stability(
    kfold_metrics, "kfold_cv", "19_kfold_average_model_stability_R2_RMSE_MAE_Bias",
)
save_table(kfold_stability, "08_kfold_average_metric_stability_data.csv")


# ## Step 8 | Grouped transferability validation
# Withhold complete seasons, treatments, or strips and refit the workflow on the remaining observations. This tests transferability to experimental conditions not represented in the corresponding training group while preserving identical held-out groups across models.

# ============================================================
# STEP 8. Leave-one-season, treatment, and strip validation
# ============================================================

group_protocols = {
    "leave_one_season_out": "season",
    "leave_one_treatment_out": "Treatment_days",
    "leave_one_strip_out": "Strip",
}

group_metrics_frames = []
group_prediction_frames = []
group_average_tables = {}

for protocol, group_column in group_protocols.items():
    plt.close("all")
    splits = SPLIT_REGISTRY[protocol]
    metrics, predictions, _ = run_validation_protocol(
        analysis_data, splits, protocol=protocol, best_parameters=BEST_PARAMETERS,
    )
    group_metrics_frames.append(metrics)
    group_prediction_frames.append(predictions)
    average_table = average_metrics_table(metrics)
    group_average_tables[protocol] = average_table
    metrics_export = append_average_accuracy_rows(metrics)

    save_table(metrics_export, f"09_{protocol}_metrics_with_average_accuracy.csv")
    save_table(average_table, f"09_{protocol}_average_metrics.csv")
    save_table(predictions, f"09_{protocol}_predictions.csv")

    print(f'{PROTOCOL_LABELS.get(protocol, protocol)} completed: {len(metrics):,} model-group metric rows')

    plt.close("all")
    for model_name in MODEL_ORDER:
        model_data = predictions[predictions["model"] == model_name]
        groups = list(dict.fromkeys(model_data["held_out_group"].tolist()))
        colors = group_color_map(groups, protocol)

        # Strip validation intentionally omits one-file-per-strip scatter plots.
        if protocol != "leave_one_strip_out":
            for group in groups:
                subset = model_data[model_data["held_out_group"] == group]
                group_label = format_group_label(protocol, group)
                plot_single_repetition_scatter(
                    subset, f"{model_display_name(model_name)}: held out {group_label}", colors[group],
                    f"{protocol}_{safe_name(model_name)}_{safe_name(group)}", observation_label=group_label,
                )

        # Strip results remain in the CSVs and integrated stability summary.
        # Do not render the large strip-position model leave-out panels.
        if protocol != "leave_one_strip_out":
            ncols = 3
            plot_repetition_grid(
                predictions, model_name, protocol, f"{protocol}_{safe_name(model_name)}_all_leaveouts_panel",
                ncols_override=ncols,
            )

    # Collective 3 x 3 panels only for season and treatment.
    if protocol in {"leave_one_season_out", "leave_one_treatment_out"}:
        plot_collective_protocol_matrix(
            predictions, protocol, f"{protocol}_all_models_3x3_panel", model_order=MODEL_ORDER,
        )

    if protocol != "leave_one_strip_out":
        plot_fold_specific_lines_panel(
            predictions, protocol, f"{protocol}_fold_specific_fitted_lines_three_model_panel",
        )

    stability = plot_average_stability(metrics, protocol, f"{protocol}_average_model_stability")
    save_table(stability, f"09_{protocol}_average_metric_stability_data.csv")

group_metrics = pd.concat(group_metrics_frames, ignore_index=True)
group_predictions = pd.concat(group_prediction_frames, ignore_index=True)


# ## Step 9 | Date-based temporal validation
# Evaluate two temporal tests: predefined harvest-date holdouts and forward-date validation using only observations collected before each evaluation date. These protocols test sensitivity to harvest-date domain shift and chronological extrapolation.

# ============================================================
# STEP 9. Selected-date and forward-date validation
# ============================================================

selected_date_splits = SPLIT_REGISTRY['targeted_date_holdout']
targeted_metrics, targeted_predictions, _ = run_validation_protocol(
    analysis_data, selected_date_splits, protocol="targeted_date_holdout", best_parameters=BEST_PARAMETERS,
)
targeted_average = average_metrics_table(targeted_metrics)
targeted_metrics_export = append_average_accuracy_rows(targeted_metrics)

forward_date_splits = SPLIT_REGISTRY['forward_date_holdout']

forward_metrics, forward_predictions, _ = run_validation_protocol(
    analysis_data, forward_date_splits, protocol="forward_date_holdout", best_parameters=BEST_PARAMETERS,
)
forward_average = average_metrics_table(forward_metrics)
forward_metrics_export = append_average_accuracy_rows(forward_metrics)

for table, filename in [
    (targeted_metrics_export, "10_targeted_date_metrics_with_average_accuracy.csv"),
    (targeted_average, "10_targeted_date_average_metrics.csv"),
    (targeted_predictions, "10_targeted_date_predictions.csv"),
    (forward_metrics_export, "10_forward_date_metrics_with_average_accuracy.csv"),
    (forward_average, "10_forward_date_average_metrics.csv"),
    (forward_predictions, "10_forward_date_predictions.csv"),
]:
    save_table(table, filename)

display(Markdown("## Selected-date validation"))
display(targeted_metrics_export.round(3))
display(Markdown("### Average selected-date performance"))
display(targeted_average.round(3))
display(Markdown("## Forward-date validation"))
display(forward_metrics_export.round(3))
display(Markdown("### Average forward-date performance"))
display(forward_average.round(3))

for protocol, predictions, metrics in [
    ("targeted_date_holdout", targeted_predictions, targeted_metrics),
    ("forward_date_holdout", forward_predictions, forward_metrics),
]:
    for model_name in MODEL_ORDER:
        model_data = predictions[predictions["model"] == model_name]
        groups = list(dict.fromkeys(model_data["held_out_group"].tolist()))
        colors = group_color_map(groups, protocol)
        for group in groups:
            subset = model_data[model_data["held_out_group"] == group]
            plot_single_repetition_scatter(
                subset, f"{model_display_name(model_name)}: {group}", colors[group],
                f"{protocol}_{safe_name(model_name)}_{safe_name(group)}", observation_label=group,
            )
        plot_repetition_grid(
            predictions, model_name, protocol, f"{protocol}_{safe_name(model_name)}_all_dates_panel",
            ncols_override=3,
        )
    plot_collective_protocol_matrix(
        predictions, protocol, f"{protocol}_all_models_3x3_panel", model_order=MODEL_ORDER,
    )
    plot_fold_specific_lines_panel(
        predictions, protocol, f"{protocol}_fold_specific_fitted_lines_three_model_panel",
    )
    stability = plot_average_stability(metrics, protocol, f"{protocol}_average_model_stability")
    save_table(stability, f"10_{protocol}_average_metric_stability_data.csv")


# ## Step 10 | Model-specific feature sensitivity
# Interpret each fitted model using a diagnostic appropriate to its structure. Ridge uses standardized coefficients with encoded season contrasts; XGBoost uses mean absolute TreeSHAP contributions for the encoded features; ANN uses the increase in RMSE after permutation, with season permuted jointly to preserve valid categories. These diagnostics quantify fitted-model reliance, not causal effects. Fold-level diagnostics are retained to distinguish stable patterns from a single fitted model.

# Final-model diagnostics and five-fold stability. No gain-to-SHAP fallback is allowed.
import xgboost as xgb
import joblib


def transformed_to_original(name):
    if name.startswith('num__'): return name[5:]
    if name.startswith('cat__season_'): return 'season'
    return name.split('__',1)[-1]


def transformed_display_name(name):
    """Display transformed columns using the same manuscript labels as the previous sensitivity figure."""
    if name.startswith('num__'):
        return pretty_feature(name[5:])
    if name.startswith('cat__season_'):
        return 'Season: ' + name.replace('cat__season_', '')
    return pretty_feature(name.split('__', 1)[-1])


def ridge_transformed_coefficients(fitted, fit_id):
    pre = fitted.named_steps['preprocessor']
    names = list(pre.get_feature_names_out())
    coefs = np.asarray(fitted.named_steps['model'].coef_).ravel()
    return pd.DataFrame({
        'fit_id': fit_id,
        'transformed_feature': names,
        'feature': [transformed_display_name(n) for n in names],
        'standardized_ridge_coefficient': coefs,
    })


def xgb_transformed_shap(fitted, evaluation, features, fit_id):
    """Mean absolute TreeSHAP at encoded-feature level, preserving separate season dummy columns."""
    pre = fitted.named_steps['preprocessor']
    est = fitted.named_steps['model']
    names = list(pre.get_feature_names_out())
    n = min(SHAP_SAMPLE_SIZE, len(evaluation))
    positions = np.sort(np.random.default_rng(RANDOM_STATE).choice(len(evaluation), n, replace=False))
    ev = evaluation.iloc[positions]
    transformed = pre.transform(ev[features])
    contrib = est.get_booster().predict(
        xgb.DMatrix(transformed, feature_names=est.get_booster().feature_names),
        pred_contribs=True, approx_contribs=False,
    )
    if contrib.ndim != 2 or contrib.shape[1] != len(names) + 1:
        raise ValueError('Unexpected XGBoost contribution dimensions.')
    pred = fitted.predict(ev[features])
    if not np.allclose(contrib.sum(axis=1), pred, rtol=2e-5, atol=.05):
        raise AssertionError('TreeSHAP contributions do not sum to the XGBoost prediction.')
    raw = np.mean(np.abs(contrib[:, :-1]), axis=0)
    total = float(np.sum(raw))
    normalized = raw / total if total > 0 else np.zeros_like(raw)
    return pd.DataFrame({
        'fit_id': fit_id,
        'transformed_feature': names,
        'feature': [transformed_display_name(nm) for nm in names],
        'mean_absolute_SHAP_raw': raw,
        # Previous manuscript figure used SHAP contributions normalized to sum to 1.
        'mean_absolute_SHAP': normalized,
        'evaluation_n': len(ev),
        'evaluation_dates': ev['Date'].nunique(),
    })


def xgb_gain_and_shap(fitted, evaluation, features, fit_id):
    pre=fitted.named_steps['preprocessor'];est=fitted.named_steps['model']
    names=list(pre.get_feature_names_out())
    raw_gain=est.get_booster().get_score(importance_type='gain')
    booster_names=est.get_booster().feature_names or [f'f{i}' for i in range(len(names))]
    gain=np.asarray([raw_gain.get(name,0.) for name in booster_names],float)
    gain_norm=gain/gain.sum() if gain.sum()>0 else np.zeros_like(gain)
    gain_table=pd.DataFrame({'fit_id':fit_id,'transformed_feature':names,
                            'feature':[transformed_to_original(n) for n in names],
                            'mean_split_gain':gain,'normalized_gain':gain_norm})
    n=min(SHAP_SAMPLE_SIZE,len(evaluation))
    positions=np.sort(np.random.default_rng(RANDOM_STATE).choice(len(evaluation),n,replace=False))
    ev=evaluation.iloc[positions]
    transformed=pre.transform(ev[features])
    contrib=est.get_booster().predict(xgb.DMatrix(transformed,feature_names=est.get_booster().feature_names),
                                    pred_contribs=True,approx_contribs=False)
    if contrib.ndim!=2 or contrib.shape[1]!=len(names)+1:
        raise ValueError('Unexpected XGBoost contribution dimensions; SHAP is not replaced with gain.')
    pred=fitted.predict(ev[features])
    if not np.allclose(contrib.sum(axis=1),pred,rtol=2e-5,atol=.05):
        raise AssertionError('TreeSHAP contributions do not sum to the model prediction.')
    # Combine signed one-hot contributions first; mean(abs(sum)) is not sum(mean(abs)).
    original_names=list(dict.fromkeys(transformed_to_original(nm) for nm in names))
    signed_groups=pd.DataFrame({feature:contrib[:,[j for j,nm in enumerate(names)
                       if transformed_to_original(nm)==feature]].sum(axis=1) for feature in original_names})
    abs_groups=signed_groups.abs()
    date_average=abs_groups.assign(Date=ev['Date'].to_numpy()).groupby('Date').mean().mean()
    shap_table=pd.DataFrame({'fit_id':fit_id,
                            'transformed_feature':[' | '.join(nm for nm in names if transformed_to_original(nm)==f)
                                                   for f in original_names],
                            'feature':original_names,'mean_absolute_SHAP':abs_groups.mean().reindex(original_names).to_numpy(),
                            'date_balanced_mean_absolute_SHAP':date_average.reindex(original_names).to_numpy(),
                            'evaluation_n':len(ev),'evaluation_dates':ev['Date'].nunique()})
    members=ev[['row_id','sample_id','Date']].copy();members['fit_id']=fit_id
    SHAP_MEMBERS.append(members)
    return gain_table,shap_table


def numeric_and_season_coefficients(fitted, fit_id):
    pre=fitted.named_steps['preprocessor'];names=list(pre.get_feature_names_out())
    coefficients=np.asarray(fitted.named_steps['model'].coef_).ravel()
    numeric=pd.DataFrame([{'fit_id':fit_id,'feature':transformed_to_original(n),
                           'coefficient_per_training_SD':float(c)}
                          for n,c in zip(names,coefficients) if n.startswith('num__')])
    contrasts=[]
    cat={n.replace('cat__season_',''):float(c) for n,c in zip(names,coefficients)
         if n.startswith('cat__season_')}
    if cat:
        reference='Fall' if 'Fall' in cat else sorted(cat)[0]
        for category,coef in cat.items():
            if category!=reference:
                contrasts.append({'fit_id':fit_id,'season':category,'reference_season':reference,
                                  'contrast_kg_ha':coef-cat[reference],
                                  'encoding':'full one-hot; no category omitted',
                                  'definition':'beta_season - beta_reference'})
    return numeric,pd.DataFrame(contrasts)


def permutation_diagnostic(fitted, evaluation, features, groups, fit_id, model):
    """Shuffle date-shared inputs by donor DATE; other inputs by rows within evaluation."""
    X=evaluation[features].copy();y=evaluation[TARGET].to_numpy(float)
    base=float(np.sqrt(mean_squared_error(y,fitted.predict(X))))
    rows=[]
    for group_name,requested in groups.items():
        columns=[c for c in requested if c in features]
        common={'fit_id':fit_id,'model':model,'feature':group_name,'selected_columns':' | '.join(columns),
                'evaluation_n':len(evaluation),'evaluation_dates':evaluation['Date'].nunique(),
                'baseline_RMSE':base}
        if not columns:
            rows.append({**common,'permutation_mean':np.nan,'permutation_sd':np.nan,
                         'status':'not_selected','permutation_unit':'not_applicable'})
            continue
        date_columns=[c for c in columns if c in WEATHER_FEATURES+PHENO_NUMERIC_FEATURES+['season']]
        if date_columns and len(date_columns)!=len(columns):
            raise ValueError('Do not jointly permute a mixture of row-level and date-level sources.')
        date_shared=bool(date_columns)
        if date_shared:
            temp=X[columns].copy();temp['_date']=evaluation['Date'].to_numpy()
            unique=temp.groupby('_date')[columns].nunique(dropna=False)
            if (unique>1).any().any():
                rows.append({**common,'permutation_mean':np.nan,'permutation_sd':np.nan,
                             'status':'skipped: values vary within date; inspect source data',
                             'permutation_unit':'date'})
                continue
            dates=np.sort(temp['_date'].unique())
            if len(dates)<2:
                rows.append({**common,'permutation_mean':np.nan,'permutation_sd':np.nan,
                             'status':'not_estimable: fewer than two dates','permutation_unit':'date'})
                continue
            representatives=temp.groupby('_date')[columns].first()
        rng=np.random.default_rng(RANDOM_STATE)
        deltas=[]
        for repetition in range(PERMUTATION_REPEATS):
            changed=X.copy()
            if date_shared:
                donors=pd.Series(rng.permutation(dates),index=dates)
                assigned=pd.Series(evaluation['Date'].to_numpy()).map(donors).to_numpy()
                changed.loc[:,columns]=representatives.loc[assigned,columns].to_numpy()
            else:
                order=rng.permutation(len(X))
                changed.loc[:,columns]=X.iloc[order][columns].to_numpy()
            deltas.append(float(np.sqrt(mean_squared_error(y,fitted.predict(changed))))-base)
        rows.append({**common,'permutation_mean':float(np.mean(deltas)),
                     'permutation_sd':float(np.std(deltas,ddof=1)) if len(deltas)>1 else 0.,
                     'status':'estimated','permutation_unit':'date' if date_shared else 'row'})
    return pd.DataFrame(rows)


def importance_axis(ax, table, value_column, model, xlabel, title, sd_column=None, signed=False):
    table=table.loc[np.isfinite(pd.to_numeric(table[value_column],errors='coerce'))].copy()
    if table.empty:
        ax.text(.5,.5,'Not estimable',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(title);return
    table=table.sort_values(value_column,key=lambda v:v.abs()) if signed else table.sort_values(value_column)
    values=table[value_column].to_numpy(float)
    errors=table[sd_column].fillna(0).to_numpy(float) if sd_column else np.zeros(len(table))
    colors=np.where(values>=0,'#2166AC','#B2182B') if signed else model_color(model)
    y=np.arange(len(table));labels=[pretty_feature(f) for f in table['feature']]
    ax.barh(y,values,xerr=errors,color=colors,height=.68,
            error_kw={'ecolor':'black','capsize':4,'linewidth':1.2})
    low=min(0.,float(np.min(values-errors)));high=max(0.,float(np.max(values+errors)))
    span=max(high-low,1.)
    for pos,value,error in zip(y,values,errors):
        right=value>=0
        if value_column == 'normalized_gain':
            label_text = f'{value:.4f}'
        elif value_column == 'standardized_ridge_coefficient':
            label_text = f'{value:,.1f}'
        else:
            label_text = f'{value:,.2f}'
        ax.annotate(label_text,(value+error if right else value-error,pos),
                    xytext=(8 if right else -8,0),textcoords='offset points',
                    ha='left' if right else 'right',va='center',fontsize=PANEL_STYLE['value'])
    ax.set_xlim(low-(.34 if signed else .20)*span,high+.29*span)
    ax.set_yticks(y,[wrap_label(label,28) for label in labels],fontsize=PANEL_STYLE['feature'])
    ax.tick_params(axis='x',labelsize=PANEL_STYLE['tick'])
    ax.set_xlabel(xlabel,fontsize=PANEL_STYLE['label']);ax.set_title(title,fontsize=PANEL_STYLE['title'],pad=15)
    ax.axvline(0,color='black',lw=1);ax.grid(axis='y',visible=False)


final_models={};SHAP_MEMBERS=[];SENSITIVITY_PROVENANCE=[]
for name in MODEL_ORDER:
    fitted=build_model_pipeline(name,GLOBAL_SELECTED_FEATURES,BEST_PARAMETERS[name])
    final_models[name]=fit_with_audit(fitted,analysis_data,GLOBAL_SELECTED_FEATURES,name,'full_data_refit','all')
    joblib.dump(final_models[name],MODEL_DIR/f'{name}_full_data_fitted_v17_final.joblib')
    SENSITIVITY_PROVENANCE.append({'fit_id':'full_data_refit','model':name,
        'training_n':len(analysis_data),'evaluation_n':len(analysis_data),
        'evaluation_role':'training-data interpretation, NOT validation',
        'parameters':parameter_signature(BEST_PARAMETERS[name]),'selected_features':' | '.join(GLOBAL_SELECTED_FEATURES)})

ridge_numeric,ridge_seasons=numeric_and_season_coefficients(final_models['Ridge'],'full_data_refit')
ridge_transformed = ridge_transformed_coefficients(final_models['Ridge'], 'full_data_refit')
xgb_gain,xgb_shap=xgb_gain_and_shap(final_models['XGBoost'],analysis_data,GLOBAL_SELECTED_FEATURES,'full_data_refit')
xgb_shap_transformed = xgb_transformed_shap(final_models['XGBoost'], analysis_data, GLOBAL_SELECTED_FEATURES, 'full_data_refit')
# Original-variable summaries remain available for source-level comparisons.
xgb_shap_original=xgb_shap.groupby('feature',as_index=False)['mean_absolute_SHAP'].sum()
xgb_gain_original=xgb_gain.groupby('feature',as_index=False)['normalized_gain'].sum()
ann_importance_table=permutation_diagnostic(final_models['ANN'],analysis_data,GLOBAL_SELECTED_FEATURES,
    {f:[f] for f in GLOBAL_SELECTED_FEATURES},'full_data_refit','ANN')
for table,name in [(ridge_numeric,'11_ridge_standardized_numeric_coefficients.csv'),
                   (ridge_seasons,'11_ridge_season_contrasts.csv'),
                   (ridge_transformed,'11_ridge_transformed_standardized_coefficients.csv'),
                   (xgb_gain,'11_xgboost_true_gain_importance.csv'),
                   (xgb_shap,'11_xgboost_true_tree_shap_importance.csv'),
                   (xgb_shap_transformed,'11_xgboost_encoded_tree_shap_importance.csv'),
                   (ann_importance_table,'11_ann_full_refit_permutation_importance.csv')]:
    save_table(table,name)

# Previous-manuscript three-panel feature-sensitivity design.
fig,axes=plt.subplots(1,3,figsize=(31,10.8))
importance_axis(axes[0],ridge_transformed,'standardized_ridge_coefficient','Ridge',
                'Standardized ridge coefficient','Ridge regression standardized coefficients',signed=True)
importance_axis(axes[1],xgb_shap_transformed,'mean_absolute_SHAP','XGBoost',
                'Mean absolute SHAP value','XGBoost feature importance')
axes[1].set_ylabel('Feature importance', fontsize=PANEL_STYLE['label'])
importance_axis(axes[2],ann_importance_table,'permutation_mean','ANN',
                'Increase in RMSE after permutation','ANN permutation importance',sd_column='permutation_sd')
# Match the earlier manuscript figure: panel letters above model-specific titles.
for ax, letter in zip(axes, ['A', 'B', 'C']):
    ax.tick_params(axis='both', labelsize=max(PANEL_STYLE['tick'], 16))
    ax.text(0.5, 1.12, letter, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=30, fontweight='bold', clip_on=False)
fig.subplots_adjust(left=.09,right=.995,bottom=.14,top=.88,wspace=.60)
save_figure(fig,'requested_feature_sensitivity_ridge_xgboost_ann_three_panel')

fig,axes=plt.subplots(1,2,figsize=(22,10))
importance_axis(axes[0],xgb_gain_original,'normalized_gain','XGBoost','Normalized split gain','XGBoost: gain')
importance_axis(axes[1],xgb_shap_original,'mean_absolute_SHAP','XGBoost',
                'Mean |TreeSHAP| (kg ha$^{-1}$)','XGBoost: TreeSHAP')
fig.subplots_adjust(left=.13,right=.99,bottom=.14,top=.93,wspace=.62)
save_figure(fig,'26_xgboost_gain_and_shap_same_fitted_model')
if not ridge_seasons.empty:
    contrasts=ridge_seasons.assign(feature=lambda t:t['season']+' versus '+t['reference_season'])
    fig,ax=plt.subplots(figsize=(11,6))
    importance_axis(ax,contrasts,'contrast_kg_ha','Ridge','Biomass contrast (kg ha$^{-1}$)',
                    'Ridge seasonal contrasts',signed=True)
    fig.tight_layout();save_figure(fig,'25b_ridge_season_contrasts_separate')

# Save standalone versions of the three principal diagnostics.
for table,value,model,label,stem,err,signed in [
    (ridge_transformed,'standardized_ridge_coefficient','Ridge','Standardized ridge coefficient',
     '25_ridge_final_standardized_coefficients',None,True),
    (xgb_shap_transformed,'mean_absolute_SHAP','XGBoost','Mean absolute SHAP value',
     '26_xgboost_final_shap_feature_importance',None,False),
    (ann_importance_table,'permutation_mean','ANN','Increase in RMSE (kg ha$^{-1}$)',
     '27_ann_permutation_importance','permutation_sd',False)]:
    fig,ax=plt.subplots(figsize=(12,8.5));importance_axis(ax,table,value,model,label,model_display_name(model),err,signed)
    fig.tight_layout();save_figure(fig,stem)

# Fold-by-fold interpretation of the EXACT five-fold models already evaluated.
fold_gain=[];fold_shap=[];fold_coeff=[];fold_contrasts=[];fold_permutation=[];source_permutation=[]
SOURCE_GROUPS={'Canopy height':HEIGHT_FEATURES,'PlanetScope':VI_FEATURES,
               'Weather + phenological timeline':WEATHER_FEATURES+PHENO_NUMERIC_FEATURES+['season']}
for fold,(train_index,test_index,group) in enumerate(SPLIT_REGISTRY['kfold_cv'],1):
    train,evaluation=analysis_data.iloc[train_index],analysis_data.iloc[test_index]
    for model in MODEL_ORDER:
        fitted=KFOLD_MODELS[(model,str(fold))]
        features=list(fitted.feature_names_in_)
        fit_id=f'kfold_{fold}'
        SENSITIVITY_PROVENANCE.append({'fit_id':fit_id,'model':model,'training_n':len(train),
            'evaluation_n':len(evaluation),'evaluation_role':'held-out fold; global tuning reused',
            'parameters':parameter_signature(BEST_PARAMETERS[model]),'selected_features':' | '.join(features)})
        pi=permutation_diagnostic(fitted,evaluation,features,{f:[f] for f in features},fit_id,model)
        pi['fold']=fold;fold_permutation.append(pi)
        spi=permutation_diagnostic(fitted,evaluation,features,SOURCE_GROUPS,fit_id,model)
        spi['fold']=fold;source_permutation.append(spi)
        if model=='Ridge':
            nc,sc=numeric_and_season_coefficients(fitted,fit_id);nc['fold']=fold;sc['fold']=fold
            fold_coeff.append(nc);fold_contrasts.append(sc)
        if model=='XGBoost':
            ga,sh=xgb_gain_and_shap(fitted,evaluation,features,fit_id);ga['fold']=fold;sh['fold']=fold
            fold_gain.append(ga);fold_shap.append(sh)
fold_gain=pd.concat(fold_gain,ignore_index=True);fold_shap=pd.concat(fold_shap,ignore_index=True)
fold_coeff=pd.concat(fold_coeff,ignore_index=True);fold_contrasts=pd.concat(fold_contrasts,ignore_index=True)
fold_permutation=pd.concat(fold_permutation,ignore_index=True);source_permutation=pd.concat(source_permutation,ignore_index=True)
for table,name in [(fold_gain,'12_xgboost_gain_by_fold.csv'),(fold_shap,'12_xgboost_shap_by_fold.csv'),
                   (fold_coeff,'12_ridge_numeric_coefficients_by_fold.csv'),(fold_contrasts,'12_ridge_season_contrasts_by_fold.csv'),
                   (fold_permutation,'12_permutation_importance_by_fold.csv'),
                   (source_permutation,'12_source_permutation_importance_by_fold.csv'),
                   (pd.DataFrame(SENSITIVITY_PROVENANCE),'12_importance_model_and_evaluation_provenance.csv'),
                   (pd.concat(SHAP_MEMBERS,ignore_index=True),'12_shap_evaluation_sample_ids.csv')]:
    save_table(table,name)

# Compare gain and SHAP computed from the same fold models, not from two different runs.
gain_fold_original=fold_gain.groupby(['fold','feature'],as_index=False)['normalized_gain'].sum()
shap_fold_original=fold_shap.groupby(['fold','feature'],as_index=False)['mean_absolute_SHAP'].sum()
GAIN_SHAP_COMPARISON=gain_fold_original.merge(shap_fold_original,on=['fold','feature'],how='outer')
GAIN_SHAP_COMPARISON['gain_rank']=GAIN_SHAP_COMPARISON.groupby('fold')['normalized_gain'].rank(ascending=False,method='min')
GAIN_SHAP_COMPARISON['shap_rank']=GAIN_SHAP_COMPARISON.groupby('fold')['mean_absolute_SHAP'].rank(ascending=False,method='min')
save_table(GAIN_SHAP_COMPARISON,'12_xgboost_gain_versus_shap_matched_folds.csv')
print('Matched-fold XGBoost gain-versus-SHAP audit exported to CSV.')

# Standard deviation here is BETWEEN folds, not a confidence interval.
for table,value,title,stem in [(gain_fold_original,'normalized_gain','Normalized gain','12a_xgboost_gain_fold_stability'),
                              (shap_fold_original,'mean_absolute_SHAP','Mean |TreeSHAP| (kg ha$^{-1}$)','12b_xgboost_shap_fold_stability')]:
    stats=table.groupby('feature')[value].agg(mean='mean',SD='std',folds_retained='count').reset_index()
    save_table(stats,stem+'.csv')
    fig,ax=plt.subplots(figsize=(12.5,9))
    importance_axis(ax,stats,'mean','XGBoost',title,'XGBoost: fold-to-fold importance',sd_column='SD')
    fig.tight_layout();save_figure(fig,stem)

SOURCE_IMPORTANCE_SUMMARY=source_permutation.groupby(['model','feature'],as_index=False).agg(
    mean=('permutation_mean','mean'),SD=('permutation_mean','std'),valid_folds=('permutation_mean','count'))
save_table(SOURCE_IMPORTANCE_SUMMARY,'12_source_importance_fold_summary.csv')
fig,axes=plt.subplots(1,3,figsize=(28,8))
for ax,model in zip(axes,MODEL_ORDER):
    importance_axis(ax,SOURCE_IMPORTANCE_SUMMARY.loc[SOURCE_IMPORTANCE_SUMMARY['model'].eq(model)],
                    'mean',model,'Increase in held-out RMSE (kg ha$^{-1}$)',model_display_name(model),sd_column='SD')
fig.subplots_adjust(left=.12,right=.99,bottom=.15,top=.92,wspace=.68)
save_figure(fig,'12c_source_permutation_fold_stability_three_model_panel')
print('Feature-sensitivity diagnostics exported.')
print('Interpretation note: removing a predictor changes the fitted conditional model and can redistribute normalized SHAP shares; tuning is optimized on RMSE, not on restoring a previous importance ranking.')
print('A change after adding records cannot be attributed to those records without a matched historical rerun.')


# ## Step 11 | Performance and stability comparison
# Combine validation metrics across protocols and models. Summary figures emphasize mean performance and between-fold/group variation while detailed prediction and audit tables are exported rather than repeatedly displayed.

all_model_metrics=pd.concat([kfold_metrics,group_metrics,targeted_metrics,forward_metrics],ignore_index=True)
all_model_predictions=pd.concat([kfold_predictions,group_predictions,targeted_predictions,forward_predictions],ignore_index=True)
all_average_metrics=average_metrics_table(all_model_metrics)
all_metrics_export=append_average_accuracy_rows(all_model_metrics)
protocol_order=PROTOCOL_ORDER;protocol_labels=PROTOCOL_LABELS
save_table(all_metrics_export,'13_all_protocol_metrics_with_average_accuracy.csv')
save_table(all_average_metrics,'13_all_protocol_average_metrics.csv')
save_table(all_model_predictions,'13_all_protocol_predictions.csv')
pooled_rows=[]
for (protocol,model),part in all_model_predictions.groupby(['protocol','model'],sort=False):
    pooled_rows.append({'protocol':protocol,'model':model,**calibration_metrics(part['observed'],part['predicted'])})
pooled_metrics=pd.DataFrame(pooled_rows)
save_table(pooled_metrics,'13_all_protocol_POOLED_metrics.csv')
fit_checks=all_model_metrics[['protocol','model','fold','held_out_group','n_train','n','train_R2','R2',
                              'train_RMSE','RMSE','RMSE_gap','train_MAE','MAE','Bias','parameter_hash']]
save_table(fit_checks,'13_training_validation_fit_checks.csv')
parameter_audit_table=pd.DataFrame(PARAMETER_AUDIT)
assert parameter_audit_table.groupby(['model','parameter'])['value'].nunique().max()==1
save_table(parameter_audit_table,'13_hyperparameter_invariance_audit.csv')
selection_audit=pd.concat(SCREENING_AUDITS,ignore_index=True)
save_table(selection_audit,'13_feature_selection_by_training_partition.csv')
selection_frequency=selection_audit.groupby(['protocol','feature'],as_index=False).agg(
    folds_evaluated=('fold','count'),folds_selected=('selected_for_model','sum'))
selection_frequency['selection_frequency']=selection_frequency['folds_selected']/selection_frequency['folds_evaluated']
save_table(selection_frequency,'13_feature_selection_frequency.csv')
comparison=baseline_metrics[['model','fold','RMSE']].rename(columns={'model':'baseline','RMSE':'baseline_RMSE'})
comparison=comparison.merge(kfold_metrics[['model','fold','RMSE']],on='fold',how='inner')
comparison['RMSE_reduction_vs_baseline']=comparison['baseline_RMSE']-comparison['RMSE']
comparison['interpretation']='Matched-fold benchmark; algorithms and source sets differ'
save_table(comparison,'13_matched_fold_baseline_comparison.csv')
STABILITY_PROTOCOL_ORDER=['kfold_cv','leave_one_strip_out','leave_one_treatment_out','leave_one_season_out','targeted_date_holdout','forward_date_holdout']
all_protocol_summary=plot_all_protocol_average_stability(all_model_metrics,STABILITY_PROTOCOL_ORDER,PROTOCOL_LABELS,
    '33c_all_protocol_model_stability_revised')
display(Markdown('### Mean validation performance'))
core_cols=['protocol','model','R2_mean','RMSE_mean','MAE_mean','Bias_mean']
display(all_average_metrics[core_cols].round(3))
print('Detailed predictions, pooled metrics, fit checks, and split audits are exported to CSV.')


# ## Step 12 | Prediction uncertainty
# Quantify residual-based 90% prediction intervals and empirical coverage. Calibration and evaluation samples are kept distinct where the workflow permits, and interval width is reported together with coverage.

# Uncertainty diagnostics: reference-band transfer and training-partition split calibration.
# Neither is labelled an independent distribution-free guarantee under global tuning/date dependence.


def residual_quantile(residuals, coverage):
    residuals=np.asarray(residuals,dtype=float)
    residuals=np.sort(residuals[np.isfinite(residuals)])
    n=len(residuals)
    if n==0: raise ValueError('No finite calibration residuals.')
    k=int(np.ceil((n+1)*coverage))
    return (float(residuals[k-1]) if k<=n else np.inf),n,k


def interval_metrics(frame):
    y=frame['observed'].to_numpy(float);lo=frame['lower'].to_numpy(float);hi=frame['upper'].to_numpy(float)
    alpha=1.-float(frame['nominal_coverage'].iloc[0])
    covered=(y>=lo)&(y<=hi)
    score=(hi-lo)+2/alpha*np.where(y<lo,lo-y,0.)+2/alpha*np.where(y>hi,y-hi,0.)
    return {'n':len(frame),'empirical_coverage':float(covered.mean()),
            'mean_width':float(np.mean(hi-lo)),'median_width':float(np.median(hi-lo)),
            'mean_interval_score':float(np.mean(score)),
            'below_interval_fraction':float(np.mean(y<lo)),
            'above_interval_fraction':float(np.mean(y>hi))}


def summarize_intervals(intervals):
    rows=[]
    for keys,frame in intervals.groupby(['method','protocol','model','nominal_coverage'],sort=False):
        method,protocol,model,coverage=keys
        rows.append(dict(method=method,protocol=protocol,model=model,nominal_coverage=coverage,
                         **interval_metrics(frame)))
    return pd.DataFrame(rows)


def plot_coverage_panel(summary, stem):
    summary=summary.loc[np.isclose(summary['nominal_coverage'],UNCERTAINTY_COVERAGE)]
    fig,axes=plt.subplots(1,3,figsize=(28,8.0))
    names=['Five-fold\nCV','Strip\nleave-out','Season\nleave-out','Treatment\nleave-out','Selected\ndate\nholdout','Forward\ndate\nholdout']
    for ax,model in zip(axes,MODEL_ORDER):
        rows=summary.loc[summary['model'].eq(model)].set_index('protocol').reindex(PROTOCOL_ORDER)
        vals=rows['empirical_coverage'].to_numpy(float);x=np.arange(len(PROTOCOL_ORDER))
        ax.bar(x,vals,color=MODEL_COLORS[model],width=.68,alpha=.92)
        ax.axhline(UNCERTAINTY_COVERAGE,color='black',ls=':',lw=2)
        for xx,value in zip(x,vals):
            if np.isfinite(value):ax.text(xx,value+.025,f'{value:.2f}',ha='center',fontsize=PANEL_STYLE['value'],
                                          bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=.4))
        ax.set_ylim(0,1.06);ax.set_xticks(x,names,fontsize=17)
        ax.set_ylabel('Empirical coverage',fontsize=21);ax.tick_params(axis='y',labelsize=17)
        ax.set_title(model_display_name(model),fontsize=23,pad=12)
    fig.legend(handles=[Line2D([0],[0],color='black',ls=':',lw=2,
                              label=f'Nominal {UNCERTAINTY_COVERAGE:.0%} coverage')],
               loc='upper center',bbox_to_anchor=(.51,.99),frameon=True,fontsize=18)
    fig.subplots_adjust(left=.06,right=.99,bottom=.24,top=.82,wspace=.28)
    save_figure(fig,stem)


def plot_interval_views(intervals, prefix):
    nominal=intervals.loc[np.isclose(intervals['nominal_coverage'],UNCERTAINTY_COVERAGE)]
    for protocol in PROTOCOL_ORDER:
        fig,axes=plt.subplots(1,3,figsize=(24,7))
        for ax,model in zip(axes,MODEL_ORDER):
            frame=nominal.loc[nominal['protocol'].eq(protocol)&nominal['model'].eq(model)].sort_values(
                ['Date','observed','sample_id']).reset_index(drop=True)
            if frame.empty: ax.set_axis_off();continue
            rank=np.arange(len(frame));color=MODEL_COLORS[model]
            ax.fill_between(rank,frame['lower'].to_numpy(float),frame['upper'].to_numpy(float),
                            color=color,alpha=.20,label=f'{UNCERTAINTY_COVERAGE:.0%} residual interval')
            ax.plot(rank,frame['center'],color=color,lw=1.3,label='Interval-model prediction')
            ax.scatter(rank,frame['observed'],s=10,color='black',alpha=.65,label='Observed biomass',zorder=3)
            met=interval_metrics(frame)
            ax.text(.02,.98,f"n = {met['n']:,}\nCoverage = {met['empirical_coverage']:.2f}\n"
                    f"Mean width = {met['mean_width']:,.0f}",transform=ax.transAxes,va='top',fontsize=17,
                    bbox=dict(facecolor='white',edgecolor='#BDBDBD',alpha=.95))
            ax.set_xlabel('Observations ordered by date and biomass',fontsize=18)
            ax.set_ylabel('Biomass (kg ha$^{-1}$)',fontsize=20)
            ax.set_title(model_display_name(model),fontsize=23);ax.tick_params(labelsize=16)
            ax.legend(loc='lower right',fontsize=14,frameon=True)
        fig.subplots_adjust(left=.065,right=.99,top=.92,bottom=.17,wspace=.30)
        save_figure(fig,f'{prefix}_{protocol}_prediction_intervals')


# A. Freeze ONE residual half-width per model using the pooled five-fold residuals.
reference_intervals=[];reference_quantiles=[]
for model in MODEL_ORDER:
    calibration=kfold_predictions.loc[kfold_predictions['model'].eq(model)].copy()
    if calibration['row_id'].duplicated().any():raise AssertionError('Duplicate OOF row in reference calibration.')
    residual=np.abs(calibration['observed']-calibration['predicted']).to_numpy()
    for coverage in UNCERTAINTY_LEVELS:
        q,n,k=residual_quantile(residual,coverage)
        reference_quantiles.append({'model':model,'nominal_coverage':coverage,'half_width':q,
            'calibration_n':n,'quantile_rank':k,'calibration_source':'pooled five-fold residuals',
            'interpretation':'reference-band stress test, not independent coverage validation'})
        for protocol in PROTOCOL_ORDER:
            part=all_model_predictions.loc[all_model_predictions['model'].eq(model)&
                                            all_model_predictions['protocol'].eq(protocol)].copy()
            part['center']=part['predicted'];part['lower']=part['center']-q;part['upper']=part['center']+q
            part['nominal_coverage']=coverage;part['half_width']=q
            part['method']='reference_kfold_residual_band'
            reference_intervals.append(part)
reference_intervals=pd.concat(reference_intervals,ignore_index=True)
reference_summary=summarize_intervals(reference_intervals)
save_table(pd.DataFrame(reference_quantiles),'34a_reference_residual_quantiles.csv')
save_table(reference_summary,'34b_reference_interval_coverage_width_score.csv')
save_table(reference_intervals.loc[np.isclose(reference_intervals['nominal_coverage'],UNCERTAINTY_COVERAGE)],
           '34c_reference_90pct_interval_predictions.csv')
plot_coverage_panel(reference_summary,'34d_uncertainty_90pct_coverage_three_model_panel')

# B. Reserve calibration dates within EACH outer training partition, using the same dates across models.
# These auxiliary interval models are not the main point-prediction fits; their centering predictions differ.
split_interval_frames=[];calibration_memberships=[];calibration_audit=[]
if RUN_SPLIT_CALIBRATION:
    for protocol in PROTOCOL_ORDER:
        for fold,(train_index,test_index,group_label) in enumerate(SPLIT_REGISTRY[protocol],1):
            outer_train=analysis_data.iloc[train_index].copy();test=analysis_data.iloc[test_index].copy()
            dates=np.sort(outer_train['Date'].unique())
            if len(dates)<3:
                raise ValueError(f'{protocol}, fold {fold}: at least three training dates are needed for date calibration.')
            n_cal_dates=min(max(1,int(np.ceil(len(dates)*CALIBRATION_DATE_FRACTION))),len(dates)-2)
            if protocol=='forward_date_holdout':
                cal_dates=dates[-n_cal_dates:]
            else:
                cal_dates=np.random.default_rng(RANDOM_STATE+fold).choice(dates,n_cal_dates,replace=False)
            calibration=outer_train.loc[outer_train['Date'].isin(cal_dates)].copy()
            proper=outer_train.loc[~outer_train['Date'].isin(cal_dates)].copy()
            if set(proper['sample_id'])&set(calibration['sample_id']) or set(test['sample_id'])&set(outer_train['sample_id']):
                raise AssertionError('Interval fit/calibration/test overlap.')
            if protocol=='forward_date_holdout' and not proper['Date'].max()<calibration['Date'].min()<test['Date'].min():
                raise AssertionError('Temporal order violated in forward interval calibration.')
            for role,frame in [('proper_training',proper),('calibration',calibration),('outer_test',test)]:
                mem=frame[['row_id','sample_id','Date']].copy();mem['protocol']=protocol;mem['fold']=str(fold);mem['role']=role
                calibration_memberships.append(mem)
            selected=screen_features(proper)['selected_features']
            for model in MODEL_ORDER:
                fitted=build_model_pipeline(model,selected,BEST_PARAMETERS[model])
                fitted=fit_with_audit(fitted,proper,selected,model,'interval_'+protocol,fold)
                residual=np.abs(calibration[TARGET].to_numpy(float)-fitted.predict(calibration[selected]))
                center=np.asarray(fitted.predict(test[selected]),float)
                common={'protocol':protocol,'fold':str(fold),'held_out_group':str(group_label),'model':model,
                        'proper_training_n':len(proper),'calibration_n':len(calibration),'testing_n':len(test),
                        'proper_training_dates':proper['Date'].nunique(),'calibration_dates':calibration['Date'].nunique(),
                        'testing_dates':test['Date'].nunique(),'calibration_test_shared_dates':len(set(calibration['Date'])&set(test['Date'])),
                        'parameter_hash':parameter_signature(BEST_PARAMETERS[model]),
                        'parameter_provenance':'full-data tuning retained; NOT selection-independent',
                        'selected_features':' | '.join(selected)}
                for coverage in UNCERTAINTY_LEVELS:
                    q,n,k=residual_quantile(residual,coverage)
                    calibration_audit.append({**common,'nominal_coverage':coverage,'half_width':q,'quantile_rank':k})
                    frame=test[['row_id','sample_id','Date','season','Treatment_days','Plot','Strip']].copy()
                    frame['Date']=frame['Date'].dt.strftime('%Y-%m-%d')
                    frame['observed']=test[TARGET].to_numpy(float);frame['center']=center
                    frame['predicted']=center;frame['lower']=center-q;frame['upper']=center+q
                    frame['half_width']=q;frame['nominal_coverage']=coverage
                    frame['protocol']=protocol;frame['model']=model;frame['fold']=str(fold);frame['held_out_group']=str(group_label)
                    frame['method']='split_calibrated_fixed_configuration'
                    split_interval_frames.append(frame)
    split_intervals=pd.concat(split_interval_frames,ignore_index=True)
    split_summary=summarize_intervals(split_intervals)
    group_rows=[]
    for (protocol,model,fold,group),frame in split_intervals.loc[
            np.isclose(split_intervals['nominal_coverage'],UNCERTAINTY_COVERAGE)].groupby(
                ['protocol','model','fold','held_out_group'],sort=False):
        group_rows.append(dict(protocol=protocol,model=model,fold=fold,held_out_group=group,**interval_metrics(frame)))
    save_table(pd.DataFrame(calibration_audit),'35a_interval_calibration_provenance.csv')
    save_table(pd.concat(calibration_memberships,ignore_index=True),'35b_fit_calibration_test_sample_ids.csv')
    save_table(split_summary,'35c_split_calibrated_coverage_width_score.csv')
    save_table(pd.DataFrame(group_rows),'35d_90pct_interval_metrics_by_fold_group.csv')
    save_table(split_intervals.loc[np.isclose(split_intervals['nominal_coverage'],UNCERTAINTY_COVERAGE)],
               '35e_split_calibrated_90pct_interval_predictions.csv')
    plot_coverage_panel(split_summary,'35f_split_calibrated_90pct_coverage_three_model_panel')
    plot_interval_views(split_intervals,'35g')

    # Reliability curves: actual coverage should track nominal levels, not just reach 90% once.
    fig,axes=plt.subplots(1,3,figsize=(24,7.2));protocol_colors=dict(zip(PROTOCOL_ORDER,distinct_colors(6)))
    for ax,model in zip(axes,MODEL_ORDER):
        for protocol in PROTOCOL_ORDER:
            part=split_summary.loc[split_summary['model'].eq(model)&split_summary['protocol'].eq(protocol)].sort_values('nominal_coverage')
            ax.plot(part['nominal_coverage'],part['empirical_coverage'],marker='o',lw=2.3,
                    color=protocol_colors[protocol],label=PROTOCOL_LABELS[protocol])
        ax.plot([.45,1],[.45,1],ls=':',color='black',lw=2,label='Nominal = empirical')
        ax.set(xlim=(.45,1),ylim=(0,1.02),xlabel='Nominal coverage',ylabel='Empirical coverage')
        ax.set_title(model_display_name(model),fontsize=23);ax.tick_params(labelsize=17)
        ax.legend(loc='lower right',fontsize=13.5,frameon=True)
    fig.subplots_adjust(left=.06,right=.99,top=.92,bottom=.15,wspace=.28)
    save_figure(fig,'35h_interval_reliability_three_model_panel')

    # Width distinguishes trivially broad intervals from useful, calibrated intervals.
    fig,axes=plt.subplots(1,3,figsize=(28,8.0))
    for ax,model in zip(axes,MODEL_ORDER):
        part=split_summary.loc[split_summary['model'].eq(model)&np.isclose(split_summary['nominal_coverage'],UNCERTAINTY_COVERAGE)]
        part=part.set_index('protocol').reindex(PROTOCOL_ORDER)
        vals=part['mean_width'].to_numpy(float);xx=np.arange(len(PROTOCOL_ORDER))
        ax.bar(xx,vals,color=MODEL_COLORS[model],alpha=.9)
        for x,v in zip(xx,vals):
            if np.isfinite(v):ax.annotate(f'{v:,.0f}',(x,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=16)
        ax.set_xticks(xx,['Five-fold\nCV','Strip\nleave-out','Season\nleave-out','Treatment\nleave-out','Selected\ndate\nholdout','Forward\ndate\nholdout'],fontsize=17)
        ax.set_ylabel('Mean interval width (kg ha$^{-1}$)',fontsize=20);ax.set_title(model_display_name(model),fontsize=23)
        ax.tick_params(axis='y',labelsize=17);ax.margins(y=.15)
    fig.subplots_adjust(left=.07,right=.99,top=.92,bottom=.24,wspace=.30)
    save_figure(fig,'35i_mean_prediction_interval_width_three_model_panel')
    display(Markdown('## Split-calibrated intervals: measured coverage and width'))
    display(split_summary.loc[np.isclose(split_summary['nominal_coverage'],UNCERTAINTY_COVERAGE)].round(3))

print('Reference-band five-fold coverage uses the residuals defining the band: it is a calibration diagnostic.')
print('Split-calibrated interval models reserve training dates for calibration, but reuse globally selected hyperparameters.')
print('Do not claim independent conformal validity: global selection, repeated dates, and domain shift remain relevant.')

# ============================================================
# STEP 10A | XGBoost TreeSHAP and gain sensitivity
# ============================================================
# Purpose:
# Calculate two complementary XGBoost importance measures from
# exactly the same fitted final model:
#
# 1. Mean absolute TreeSHAP:
#    Average magnitude of each predictor's contribution to biomass
#    predictions. Values are reported in biomass prediction units.
#
# 2. Normalized gain:
#    Relative improvement in the XGBoost objective contributed by
#    tree splits involving each predictor.
#
# No model refitting is performed in this section.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ------------------------------------------------------------
# Retrieve the exact fitted XGBoost pipeline
# ------------------------------------------------------------

xgb_pipeline = final_models["XGBoost"]

preprocessor = xgb_pipeline.named_steps["preprocessor"]
xgb_model = xgb_pipeline.named_steps["model"]
booster = xgb_model.get_booster()

# ------------------------------------------------------------
# Transform the same final modeling dataset
# ------------------------------------------------------------

X_original = analysis_data[GLOBAL_SELECTED_FEATURES].copy()
X_transformed = preprocessor.transform(X_original)

feature_names = list(
    preprocessor.get_feature_names_out()
)

# Convert sparse matrix if required
if hasattr(X_transformed, "toarray"):
    X_transformed_dense = X_transformed.toarray()
else:
    X_transformed_dense = np.asarray(X_transformed)

# ------------------------------------------------------------
# Consistent manuscript display names
# ------------------------------------------------------------

def encoded_display_name(name):

    name = str(name)

    if name.startswith("num__"):
        raw = name.replace("num__", "", 1)
        return pretty_feature(raw)

    if name.startswith("cat__season_"):
        level = name.replace("cat__season_", "", 1)
        return f"Season: {level}"

    if name.startswith("cat__"):
        raw = name.replace("cat__", "", 1)

        if raw.startswith("season_"):
            return "Season: " + raw.replace("season_", "", 1)

        return pretty_feature(raw)

    return pretty_feature(name)


display_names = [
    encoded_display_name(f)
    for f in feature_names
]

print(f"Evaluation observations: {len(X_original):,}")
print(f"Encoded XGBoost predictors: {len(feature_names)}")

# ============================================================
# Exact TreeSHAP contributions
# ============================================================

dmatrix = xgb.DMatrix(
    X_transformed_dense
)

shap_matrix = booster.predict(
    dmatrix,
    pred_contribs=True,
    approx_contribs=False
)

# Last column is the XGBoost expected-value/bias term
shap_values = shap_matrix[:, :-1]
shap_base_value = shap_matrix[:, -1]

if shap_values.shape[1] != len(feature_names):
    raise ValueError(
        "Number of TreeSHAP columns does not match "
        "the transformed feature count."
    )

# ------------------------------------------------------------
# Verify SHAP additivity
# ------------------------------------------------------------

model_predictions = xgb_pipeline.predict(
    X_original
)

reconstructed_predictions = (
    shap_values.sum(axis=1)
    + shap_base_value
)

if not np.allclose(
    model_predictions,
    reconstructed_predictions,
    rtol=2e-5,
    atol=0.05
):
    raise AssertionError(
        "TreeSHAP contributions do not reconstruct "
        "the fitted XGBoost predictions."
    )

# ------------------------------------------------------------
# Global mean absolute TreeSHAP
# ------------------------------------------------------------

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

xgb_shap_table = pd.DataFrame({
    "encoded_feature": feature_names,
    "Predictor": display_names,
    "Mean_absolute_TreeSHAP": mean_abs_shap
})

xgb_shap_table = (
    xgb_shap_table
    .sort_values(
        "Mean_absolute_TreeSHAP",
        ascending=False
    )
    .reset_index(drop=True)
)

display(xgb_shap_table)

save_table(
    xgb_shap_table,
    "11b_xgboost_mean_absolute_treeshap.csv"
)
# ============================================================
# FIGURE | XGBoost global TreeSHAP importance
# ============================================================

plot_df = xgb_shap_table.copy()

fig, ax = plt.subplots(
    figsize=(11.5, 8.5)
)

y = np.arange(len(plot_df))

bars = ax.barh(
    y,
    plot_df["Mean_absolute_TreeSHAP"],
    color=MODEL_COLORS["XGBoost"],
    alpha=0.92
)

ax.set_yticks(
    y,
    plot_df["Predictor"]
)

ax.invert_yaxis()

ax.set_xlabel(
    r"Mean absolute TreeSHAP value (kg DM ha$^{-1}$)",
    fontsize=21
)

ax.set_ylabel(
    "Predictor",
    fontsize=21
)

ax.set_title(
    "XGBoost global feature importance",
    fontsize=23,
    pad=14
)

ax.tick_params(
    axis="both",
    labelsize=18
)

ax.grid(
    axis="x",
    alpha=0.18
)

ax.grid(
    axis="y",
    visible=False
)

# ------------------------------------------------------------
# Numerical values
# ------------------------------------------------------------

xmax = plot_df["Mean_absolute_TreeSHAP"].max()

for bar, value in zip(
    bars,
    plot_df["Mean_absolute_TreeSHAP"]
):

    ax.text(
        value + xmax * 0.012,
        bar.get_y() + bar.get_height() / 2,
        f"{value:,.1f}",
        va="center",
        ha="left",
        fontsize=17
    )

ax.set_xlim(
    0,
    xmax * 1.16
)

fig.tight_layout()

save_figure(
    fig,
    "11b_xgboost_mean_absolute_treeshap_global"
)
# ============================================================
# FIGURE | XGBoost global TreeSHAP importance
# ============================================================

plot_df = xgb_shap_table.copy()

fig, ax = plt.subplots(
    figsize=(11.5, 8.5)
)

y = np.arange(len(plot_df))

bars = ax.barh(
    y,
    plot_df["Mean_absolute_TreeSHAP"],
    color=MODEL_COLORS["XGBoost"],
    alpha=0.92
)

ax.set_yticks(
    y,
    plot_df["Predictor"]
)

ax.invert_yaxis()

ax.set_xlabel(
    r"Mean absolute TreeSHAP value (kg DM ha$^{-1}$)",
    fontsize=21
)

ax.set_ylabel(
    "Predictor",
    fontsize=21
)

ax.set_title(
    "XGBoost global feature importance",
    fontsize=23,
    pad=14
)

ax.tick_params(
    axis="both",
    labelsize=18
)

ax.grid(
    axis="x",
    alpha=0.18
)

ax.grid(
    axis="y",
    visible=False
)

# ------------------------------------------------------------
# Numerical values
# ------------------------------------------------------------

xmax = plot_df["Mean_absolute_TreeSHAP"].max()

for bar, value in zip(
    bars,
    plot_df["Mean_absolute_TreeSHAP"]
):

    ax.text(
        value + xmax * 0.012,
        bar.get_y() + bar.get_height() / 2,
        f"{value:,.1f}",
        va="center",
        ha="left",
        fontsize=17
    )

ax.set_xlim(
    0,
    xmax * 1.16
)

fig.tight_layout()

save_figure(
    fig,
    "11b_xgboost_mean_absolute_treeshap_global"
)
# ============================================================
# FIGURE | Three-model feature sensitivity
# Panel A: Ridge standardized coefficients
# Panel B: XGBoost mean absolute TreeSHAP
# Panel C: ANN permutation importance
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(31, 10.8)
)

# ------------------------------------------------------------
# PANEL A | Ridge
# ------------------------------------------------------------

importance_axis(
    axes[0],
    ridge_transformed,
    "standardized_ridge_coefficient",
    "Ridge",
    "Standardized ridge coefficient",
    "Ridge regression standardized coefficients",
    signed=True
)

# ------------------------------------------------------------
# PANEL B | XGBoost TreeSHAP
# ------------------------------------------------------------

shap_panel = xgb_shap_table.rename(
    columns={
        "Predictor": "feature",
        "Mean_absolute_TreeSHAP": "mean_absolute_SHAP"
    }
)

importance_axis(
    axes[1],
    shap_panel,
    "mean_absolute_SHAP",
    "XGBoost",
    r"Mean absolute TreeSHAP value (kg DM ha$^{-1}$)",
    "XGBoost feature importance"
)

axes[1].set_ylabel(
    "Predictor",
    fontsize=PANEL_STYLE["label"]
)

# ------------------------------------------------------------
# PANEL C | ANN
# ------------------------------------------------------------

importance_axis(
    axes[2],
    ann_importance_table,
    "permutation_mean",
    "ANN",
    r"Increase in RMSE after permutation (kg DM ha$^{-1}$)",
    "ANN permutation importance",
    sd_column="permutation_sd"
)

# ------------------------------------------------------------
# Panel labels
# ------------------------------------------------------------

for ax, letter in zip(
    axes,
    ["A", "B", "C"]
):

    ax.text(
        0.5,
        1.12,
        letter,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=30,
        fontweight="bold",
        clip_on=False
    )

    ax.tick_params(
        axis="both",
        labelsize=max(
            PANEL_STYLE["tick"],
            16
        )
    )

fig.subplots_adjust(
    left=0.09,
    right=0.995,
    bottom=0.14,
    top=0.88,
    wspace=0.60
)

save_figure(
    fig,
    "11c_feature_sensitivity_three_panel_treeshap"
)
# ============================================================
# APPENDIX FIGURE | XGBoost TreeSHAP beeswarm
# ============================================================

# Number of predictors displayed
TOP_N_SHAP = min(
    12,
    len(feature_names)
)

top_features = (
    xgb_shap_table
    .head(TOP_N_SHAP)
    ["encoded_feature"]
    .tolist()
)

feature_index = {
    name: i
    for i, name in enumerate(feature_names)
}

top_indices = [
    feature_index[f]
    for f in top_features
]

top_labels = [
    encoded_display_name(f)
    for f in top_features
]

# ------------------------------------------------------------
# Draw observations for clarity
# ------------------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

MAX_BEESWARM_POINTS = 800

if len(X_transformed_dense) > MAX_BEESWARM_POINTS:

    sample_idx = np.sort(
        rng.choice(
            len(X_transformed_dense),
            MAX_BEESWARM_POINTS,
            replace=False
        )
    )

else:

    sample_idx = np.arange(
        len(X_transformed_dense)
    )


fig, ax = plt.subplots(
    figsize=(12.5, 9.5)
)

# ------------------------------------------------------------
# Plot each feature
# ------------------------------------------------------------

for row, (
    idx,
    label
) in enumerate(
    zip(top_indices, top_labels)
):

    shap_x = shap_values[
        sample_idx,
        idx
    ]

    feature_value = X_transformed_dense[
        sample_idx,
        idx
    ].astype(float)

    # Robust color normalization
    lo, hi = np.nanpercentile(
        feature_value,
        [5, 95]
    )

    if np.isclose(lo, hi):
        lo = np.nanmin(feature_value)
        hi = np.nanmax(feature_value)

    if np.isclose(lo, hi):
        hi = lo + 1

    scaled_value = np.clip(
        (feature_value - lo)
        / (hi - lo),
        0,
        1
    )

    # Small deterministic vertical jitter
    jitter = rng.normal(
        loc=0,
        scale=0.085,
        size=len(shap_x)
    )

    ax.scatter(
        shap_x,
        row + jitter,
        c=scaled_value,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=24,
        alpha=0.68,
        edgecolors="none"
    )


# ------------------------------------------------------------
# Zero contribution reference
# ------------------------------------------------------------

ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.3
)

ax.set_yticks(
    np.arange(TOP_N_SHAP),
    top_labels
)

ax.invert_yaxis()

ax.set_xlabel(
    r"TreeSHAP contribution to predicted biomass (kg DM ha$^{-1}$)",
    fontsize=20
)

ax.set_ylabel(
    "Predictor",
    fontsize=20
)

ax.set_title(
    "XGBoost TreeSHAP distribution",
    fontsize=23,
    pad=14
)

ax.tick_params(
    axis="both",
    labelsize=17
)

ax.grid(
    axis="x",
    alpha=0.18
)

ax.grid(
    axis="y",
    visible=False
)

# ------------------------------------------------------------
# Shared colorbar
# ------------------------------------------------------------

sm = ScalarMappable(
    norm=Normalize(
        vmin=0,
        vmax=1
    ),
    cmap="coolwarm"
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    pad=0.02,
    fraction=0.035
)

cbar.set_label(
    "Relative feature value",
    fontsize=18
)

cbar.set_ticks(
    [0, 1]
)

cbar.set_ticklabels(
    ["Low", "High"]
)

cbar.ax.tick_params(
    labelsize=16
)

fig.tight_layout()

save_figure(
    fig,
    "Appendix_XGBoost_TreeSHAP_beeswarm"
)
# ============================================================
# SUPPLEMENTARY | XGBoost normalized gain importance
# ============================================================

raw_gain = booster.get_score(
    importance_type="gain"
)

# Booster may use f0, f1, ... when fitting on transformed arrays
gain_values = []

for i, transformed_name in enumerate(
    feature_names
):

    possible_keys = [
        transformed_name,
        f"f{i}"
    ]

    value = 0.0

    for key in possible_keys:

        if key in raw_gain:
            value = raw_gain[key]
            break

    gain_values.append(value)


gain_values = np.asarray(
    gain_values,
    dtype=float
)

if gain_values.sum() > 0:

    normalized_gain = (
        gain_values
        / gain_values.sum()
    )

else:

    normalized_gain = np.zeros_like(
        gain_values
    )


xgb_gain_table = pd.DataFrame({
    "encoded_feature": feature_names,
    "Predictor": display_names,
    "Mean_split_gain": gain_values,
    "Normalized_gain": normalized_gain
})

xgb_gain_table = (
    xgb_gain_table
    .sort_values(
        "Normalized_gain",
        ascending=False
    )
    .reset_index(drop=True)
)

display(xgb_gain_table)

save_table(
    xgb_gain_table,
    "Supplementary_XGBoost_normalized_gain.csv"
)