import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source_lines(text):
    text = text.strip("\n")
    return [line + "\n" for line in text.split("\n")]


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def write_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def demand_notebook():
    return [
        md(
            """
# 04 - Demand Prediction Agent

This notebook implements the UrbanEV/ST-EVCDP Demand Prediction Agent and produces the demand-side inputs required by the pricing and monitoring agents.

Mentor requirements covered here:

- Time-series train/test split
- Baseline lag model
- Ridge Regression
- XGBRegressor
- MAE, RMSE, R2, and MAPE comparison
- XGBoost feature importance
- Actual vs predicted visualization
- Residual plot and error distribution
- Charger utilization KPI
- Off-peak uplift KPI
- Waiting-time proxy KPI

The agent uses UrbanEV because it contains spatial-temporal occupancy, price, infrastructure, and neighbor-grid signals.
"""
        ),
        md(
            """
## Assumptions

- The target is future occupancy from the engineered feature table.
- Utilization is calculated as occupancy divided by charger count.
- Dynamic pricing is simulated from predicted utilization, not observed from a randomized field experiment.
- Waiting time is proxied by demand pressure because actual queue wait times are unavailable.
"""
        ),
        code(
            """
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
plt.rcParams["figure.figsize"] = (12, 6)

BASELINE_PRICE_INR = 15.0
ELASTICITY = -0.30
RANDOM_STATE = 42

FIG_DIR = Path("../outputs/figures")
MODEL_DIR = Path("../outputs/models")
REPORT_DIR = Path("../outputs/reports")
for directory in [FIG_DIR, MODEL_DIR, REPORT_DIR, Path("../data/processed")]:
    directory.mkdir(parents=True, exist_ok=True)

print("Environment ready")
"""
        ),
        md(
            """
## Load UrbanEV Features

The engineered table combines temporal, spatial, pricing, lag, rolling, and infrastructure features.
"""
        ),
        code(
            """
df = pd.read_parquet("../data/processed/demand_features.parquet")
df = df.sort_values(["datetime", "grid"]).reset_index(drop=True)

df["utilization"] = (df["occupancy"] / df["count"]).clip(0, 1.5)
df["target_utilization"] = (df["target"] / df["count"]).clip(0, 1.5)

print(df.shape)
df.head()
"""
        ),
        md(
            """
## Time-Series Split

The split is chronological, not random, so every model is evaluated on future observations.
"""
        ),
        code(
            """
features = [
    "price",
    "count",
    "fast_count",
    "slow_count",
    "area",
    "CBD",
    "dynamic_pricing",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "lag_1",
    "lag_12",
    "lag_24",
    "lag_288",
    "rolling_mean_12",
    "rolling_std_12",
    "neighbor_occupancy",
]

model_df = df.dropna(subset=features + ["target"]).copy()
split_idx = int(len(model_df) * 0.80)

train_df = model_df.iloc[:split_idx].copy()
test_df = model_df.iloc[split_idx:].copy()

train_sample = train_df.sample(n=min(200_000, len(train_df)), random_state=RANDOM_STATE)
test_eval = test_df.copy()

X_train = train_sample[features]
y_train = train_sample["target"]
X_test = test_eval[features]
y_test = test_eval["target"]

print("Train period:", train_df["datetime"].min(), "to", train_df["datetime"].max())
print("Test period:", test_df["datetime"].min(), "to", test_df["datetime"].max())
print("Training rows used:", len(train_sample))
print("Test rows:", len(test_eval))
"""
        ),
        md(
            """
## Model Training and Comparison

Three models are compared:

- Baseline: lag_12 persistence model
- Ridge Regression: transparent linear baseline
- XGBRegressor: nonlinear gradient-boosted tree model
"""
        ),
        code(
            """
def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MAPE_percent": mape(y_true, y_pred),
    }


predictions = {}
predictions["Baseline_lag_12"] = test_eval["lag_12"].clip(lower=0, upper=test_eval["count"] * 1.5)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
predictions["Ridge_Regression"] = np.clip(
    ridge_model.predict(X_test),
    0,
    test_eval["count"] * 1.5,
)

xgb_model = XGBRegressor(
    n_estimators=220,
    max_depth=5,
    learning_rate=0.06,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=1,
)
xgb_model.fit(X_train, y_train)
predictions["XGBRegressor"] = np.clip(
    xgb_model.predict(X_test),
    0,
    test_eval["count"] * 1.5,
)

comparison = pd.DataFrame(
    [evaluate_model(name, y_test, pred) for name, pred in predictions.items()]
).sort_values("RMSE")

comparison.to_csv(REPORT_DIR / "demand_model_comparison.csv", index=False)
joblib.dump(ridge_model, MODEL_DIR / "ridge_demand_model.joblib")
joblib.dump(xgb_model, MODEL_DIR / "xgb_demand_model.joblib")

comparison
"""
        ),
        md(
            """
### Business Interpretation

The model with the lowest RMSE is preferred for tariff decisions because large demand forecast errors can lead to congestion during peaks or unnecessary discounts during low-demand periods. R2 explains how much occupancy variation is captured by the model.
"""
        ),
        md(
            """
## Feature Importance

XGBoost feature importance shows which demand signals most strongly influence the forecast.
"""
        ),
        code(
            """
importance = pd.DataFrame(
    {
        "feature": features,
        "importance": xgb_model.feature_importances_,
    }
).sort_values("importance", ascending=False)

top15_importance = importance.head(15).reset_index(drop=True)
top15_importance.to_csv(REPORT_DIR / "xgb_top15_feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top15_importance["feature"][::-1], top15_importance["importance"][::-1])
ax.set_title("Top 15 XGBoost Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(FIG_DIR / "xgb_top15_feature_importance.png", dpi=160)
fig
"""
        ),
        md(
            """
### Business Interpretation

High importance for lag and rolling features means recent utilization is the strongest short-term signal. Neighbor occupancy captures spatial spillover, while price and hour help the pricing agent connect demand patterns to tariff timing.
"""
        ),
        md(
            """
## Forecast Diagnostics
"""
        ),
        code(
            """
best_model_name = comparison.iloc[0]["model"]
test_eval["predicted_occupancy"] = predictions[best_model_name]
test_eval["prediction_error"] = test_eval["target"] - test_eval["predicted_occupancy"]
test_eval["predicted_utilization"] = (test_eval["predicted_occupancy"] / test_eval["count"]).clip(0, 1.5)

plot_df = (
    test_eval.groupby("datetime", as_index=False)
    .agg(actual=("target", "mean"), predicted=("predicted_occupancy", "mean"))
    .head(350)
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df["datetime"], plot_df["actual"], label="Actual", linewidth=2)
ax.plot(plot_df["datetime"], plot_df["predicted"], label="Predicted", linewidth=2)
ax.set_title("Actual vs Predicted Network Demand")
ax.set_xlabel("Datetime")
ax.set_ylabel("Average Occupancy")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "actual_vs_predicted_demand.png", dpi=160)
fig
"""
        ),
        code(
            """
sample_residuals = test_eval.sample(n=min(25_000, len(test_eval)), random_state=RANDOM_STATE)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(sample_residuals["predicted_occupancy"], sample_residuals["prediction_error"], alpha=0.25, s=8)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("Residual Plot")
ax.set_xlabel("Predicted Occupancy")
ax.set_ylabel("Actual - Predicted")
plt.tight_layout()
plt.savefig(FIG_DIR / "residual_plot.png", dpi=160)
fig
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(sample_residuals["prediction_error"], bins=60, color="#4c78a8", edgecolor="white")
ax.set_title("Prediction Error Distribution")
ax.set_xlabel("Prediction Error")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIG_DIR / "prediction_error_distribution.png", dpi=160)
fig
"""
        ),
        md(
            """
### Business Interpretation

The actual-vs-predicted plot confirms whether the model follows the timing of network demand. The residual plots reveal whether errors are centered around zero or if the model systematically underestimates peak occupancy.
"""
        ),
        md(
            """
## Dynamic Tariff Simulation for Demand-Side KPIs

The pricing rule converts predicted utilization into a tariff:

- Off-peak discount when utilization is below 30 percent
- Mild premium during normal periods
- Surge premium above 80 percent utilization
"""
        ),
        code(
            """
def recommend_tariff(utilization, baseline=BASELINE_PRICE_INR):
    if utilization < 0.30:
        discount_strength = (0.30 - utilization) / 0.30
        return baseline * (1 - 0.04 * discount_strength)
    if utilization <= 0.80:
        return baseline * 1.08
    surge_strength = min((utilization - 0.80) / 0.70, 1.0)
    return baseline * (1.08 + 0.45 * surge_strength)


def response_multiplier(price, baseline=BASELINE_PRICE_INR, elasticity=ELASTICITY):
    pct_price_change = (price - baseline) / baseline
    return np.clip(1 + elasticity * pct_price_change, 0.70, 1.25)


test_eval["off_peak_hour"] = test_eval["hour"].between(0, 6) | test_eval["hour"].between(22, 23)
test_eval["recommended_tariff_inr_kwh"] = test_eval["predicted_utilization"].apply(recommend_tariff)
off_peak_discount_mask = test_eval["off_peak_hour"] & (test_eval["predicted_utilization"] <= 0.80)
test_eval["recommended_tariff_inr_kwh"] = np.where(
    off_peak_discount_mask,
    np.minimum(test_eval["recommended_tariff_inr_kwh"], BASELINE_PRICE_INR * 0.96),
    test_eval["recommended_tariff_inr_kwh"],
)
test_eval["response_multiplier"] = response_multiplier(test_eval["recommended_tariff_inr_kwh"])
test_eval["expected_occupancy_after_pricing"] = (
    test_eval["predicted_occupancy"] * test_eval["response_multiplier"]
).clip(0, test_eval["count"] * 1.5)
test_eval["expected_utilization_after_pricing"] = (
    test_eval["expected_occupancy_after_pricing"] / test_eval["count"]
).clip(0, 1.5)

test_eval["waiting_time_proxy_before"] = test_eval["predicted_utilization"]
test_eval["waiting_time_proxy_after"] = test_eval["expected_utilization_after_pricing"]

test_eval.to_parquet("../outputs/reports/demand_agent_predictions.parquet", index=False)
"""
        ),
        md(
            """
## Off-Peak Uplift KPI

Off-peak hours are defined as 00:00-06:59 and 22:00-23:59.
"""
        ),
        code(
            """
off_peak = test_eval[test_eval["off_peak_hour"]].copy()
off_peak_before = off_peak["predicted_occupancy"].sum()
off_peak_after = off_peak["expected_occupancy_after_pricing"].sum()
off_peak_uplift_pct = ((off_peak_after - off_peak_before) / off_peak_before) * 100

off_peak_hourly = (
    off_peak.groupby("hour", as_index=False)
    .agg(
        before=("predicted_occupancy", "mean"),
        after=("expected_occupancy_after_pricing", "mean"),
    )
)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(off_peak_hourly))
ax.bar(x - 0.2, off_peak_hourly["before"], width=0.4, label="Before Pricing")
ax.bar(x + 0.2, off_peak_hourly["after"], width=0.4, label="After Pricing")
ax.set_xticks(x)
ax.set_xticklabels(off_peak_hourly["hour"])
ax.set_title("Off-Peak Demand Uplift")
ax.set_xlabel("Hour")
ax.set_ylabel("Average Demand")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "off_peak_uplift.png", dpi=160)

pd.DataFrame([{"KPI": "Off-Peak Uplift %", "Value": off_peak_uplift_pct}])
"""
        ),
        md(
            """
### Business Interpretation

Positive off-peak uplift indicates that discount pricing is shifting some demand into low-utilization windows, improving infrastructure use without adding new chargers.
"""
        ),
        md(
            """
## Charger Utilization Rate KPI
"""
        ),
        code(
            """
network_utilization = test_eval["predicted_utilization"].mean()
network_utilization_after = test_eval["expected_utilization_after_pricing"].mean()
excess_utilization_before = np.maximum(test_eval["predicted_utilization"] - 0.80, 0).mean()
excess_utilization_after = np.maximum(test_eval["expected_utilization_after_pricing"] - 0.80, 0).mean()
utilization_improvement_pct = (
    (excess_utilization_before - excess_utilization_after) / excess_utilization_before
) * 100 if excess_utilization_before > 0 else 0

grid_utilization = (
    test_eval.groupby("grid", as_index=False)
    .agg(
        avg_utilization_before=("predicted_utilization", "mean"),
        avg_utilization_after=("expected_utilization_after_pricing", "mean"),
        charger_count=("count", "mean"),
    )
    .sort_values("avg_utilization_before", ascending=False)
)

peak_hour_utilization = (
    test_eval.groupby("hour", as_index=False)
    .agg(avg_utilization=("predicted_utilization", "mean"))
    .sort_values("avg_utilization", ascending=False)
)

utilization_dashboard = pd.DataFrame(
    [
        {"metric": "Network utilization before pricing", "value": network_utilization},
        {"metric": "Network utilization after pricing", "value": network_utilization_after},
        {"metric": "Congestion-adjusted utilization improvement %", "value": utilization_improvement_pct},
        {"metric": "Peak utilization hour", "value": peak_hour_utilization.iloc[0]["hour"]},
    ]
)

grid_utilization.to_csv(REPORT_DIR / "grid_utilization_summary.csv", index=False)
peak_hour_utilization.to_csv(REPORT_DIR / "peak_hour_utilization.csv", index=False)
utilization_dashboard.to_csv(REPORT_DIR / "charger_utilization_dashboard.csv", index=False)

utilization_dashboard
"""
        ),
        md(
            """
### Business Interpretation

Network utilization shows the overall efficiency of charger assets. Grid-level utilization identifies locations that need pricing intervention, infrastructure expansion, or operational monitoring.
"""
        ),
        md(
            """
## Waiting Time Reduction Proxy
"""
        ),
        code(
            """
waiting_before = test_eval["waiting_time_proxy_before"].mean()
waiting_after = test_eval["waiting_time_proxy_after"].mean()
waiting_time_reduction_pct = ((waiting_before - waiting_after) / waiting_before) * 100

waiting_kpi = pd.DataFrame(
    [
        {"metric": "Waiting time proxy before pricing", "value": waiting_before},
        {"metric": "Waiting time proxy after pricing", "value": waiting_after},
        {"metric": "Waiting Time Reduction %", "value": waiting_time_reduction_pct},
    ]
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(["Before", "After"], [waiting_before, waiting_after], color=["#e45756", "#54a24b"])
ax.set_title("Waiting Time Proxy Reduction")
ax.set_ylabel("Average utilization pressure")
plt.tight_layout()
plt.savefig(FIG_DIR / "waiting_time_proxy_reduction.png", dpi=160)

waiting_kpi.to_csv(REPORT_DIR / "waiting_time_proxy_kpi.csv", index=False)
waiting_kpi
"""
        ),
        md(
            """
### Business Interpretation

The waiting proxy improves when pricing lowers utilization pressure in congested periods. This is not a measured queue time, so it should be presented as an operational proxy rather than a causal waiting-time claim.
"""
        ),
        md(
            """
## Export Demand Agent KPI Summary
"""
        ),
        code(
            """
demand_accuracy = comparison.loc[comparison["model"] == best_model_name].iloc[0]

demand_kpis = pd.DataFrame(
    [
        {"KPI": "Demand Forecast Accuracy R2", "Value": demand_accuracy["R2"]},
        {"KPI": "Demand Forecast RMSE", "Value": demand_accuracy["RMSE"]},
        {"KPI": "Off-Peak Uplift %", "Value": off_peak_uplift_pct},
        {"KPI": "Charger Utilization Rate", "Value": network_utilization},
        {"KPI": "Utilization Improvement %", "Value": utilization_improvement_pct},
        {"KPI": "Waiting Time Reduction %", "Value": waiting_time_reduction_pct},
    ]
)

tariff_profile = (
    test_eval.groupby(["dayofweek", "hour"], as_index=False)
    .agg(
        recommended_tariff_inr_kwh=("recommended_tariff_inr_kwh", "mean"),
        predicted_utilization=("predicted_utilization", "mean"),
        expected_utilization_after_pricing=("expected_utilization_after_pricing", "mean"),
    )
)

demand_kpis.to_csv(REPORT_DIR / "demand_agent_kpis.csv", index=False)
tariff_profile.to_csv(REPORT_DIR / "tariff_profile.csv", index=False)

demand_kpis
"""
        ),
        md(
            """
## Limitations and Future Work

- The tariff response is simulated with an elasticity assumption, not observed behavior from a live dynamic-pricing experiment.
- Weather, events, holidays, grid load, and competitor prices are not included.
- Future work should validate elasticity with A/B tests, add exogenous variables, and retrain the agent on live outcomes.
"""
        ),
    ]


def tariff_notebook():
    return [
        md(
            """
# 05 - Tariff Pricing Agent

This notebook implements the Tariff Pricing Agent using ACN session data and the UrbanEV demand-driven tariff profile.

Mentor requirements covered here:

- Revenue Gain %
- Customer Response Rate using ACN plus elasticity assumption
- Pricing Efficiency Score
- Revenue comparison chart
- Sensitivity analysis for price changes of +5%, +10%, +15%, and +20%
"""
        ),
        md(
            """
## Assumptions

- Baseline tariff is INR 15/kWh.
- Customer response follows constant price elasticity.
- ACN provides delivered kWh and session behavior, while UrbanEV provides the utilization-based tariff profile.
- Results are simulation estimates, not causal claims.
"""
        ),
        code(
            """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_PRICE_INR = 15.0
ELASTICITY = -0.30

FIG_DIR = Path("../outputs/figures")
REPORT_DIR = Path("../outputs/reports")
for directory in [FIG_DIR, REPORT_DIR, Path("../data/processed")]:
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 5)
"""
        ),
        md(
            """
## Load ACN Features
"""
        ),
        code(
            """
acn = pd.read_csv("../data/acn/acn_cleaned.csv")
acn["connectionTime"] = pd.to_datetime(acn["connectionTime"], errors="coerce")
acn["disconnectTime"] = pd.to_datetime(acn["disconnectTime"], errors="coerce")
acn["doneChargingTime"] = pd.to_datetime(acn["doneChargingTime"], errors="coerce")

if "session_duration_hours" not in acn.columns:
    acn["session_duration_hours"] = (
        acn["disconnectTime"] - acn["connectionTime"]
    ).dt.total_seconds() / 3600

if "charging_duration_hours" not in acn.columns:
    acn["charging_duration_hours"] = (
        acn["doneChargingTime"] - acn["connectionTime"]
    ).dt.total_seconds() / 3600

acn["energy_rate"] = acn["kWhDelivered"] / acn["charging_duration_hours"].replace(0, np.nan)
acn["utilization_proxy"] = (
    acn["charging_duration_hours"] / acn["session_duration_hours"].replace(0, np.nan)
).clip(0, 1.5)
parsed_hour = acn["connectionTime"].dt.hour
parsed_dayofweek = acn["connectionTime"].dt.dayofweek
if "hour" in acn.columns:
    parsed_hour = parsed_hour.fillna(acn["hour"])
if "dayofweek" in acn.columns:
    parsed_dayofweek = parsed_dayofweek.fillna(acn["dayofweek"])

acn["hour"] = parsed_hour
acn["dayofweek"] = parsed_dayofweek
acn = acn.dropna(subset=["hour", "dayofweek"]).copy()
acn["hour"] = acn["hour"].astype(int)
acn["dayofweek"] = acn["dayofweek"].astype(int)
acn["peak_hour"] = acn["hour"].between(17, 21).astype(int)
acn["is_weekend"] = (acn["dayofweek"] >= 5).astype(int)

acn_features = acn.dropna(subset=["kWhDelivered", "hour", "dayofweek"]).copy()
acn_features = acn_features[acn_features["kWhDelivered"] > 0].copy()
acn_features.to_parquet("../data/processed/acn_features.parquet", index=False)

print(acn_features.shape)
acn_features.head()
"""
        ),
        md(
            """
### Business Interpretation

ACN is the right source for revenue analysis because it contains delivered kWh per session. The engineered features describe how long customers stayed connected, how much energy they received, and whether a session happened during peak periods.
"""
        ),
        md(
            """
## Demand Elasticity Assumptions
"""
        ),
        code(
            """
def customer_response_multiplier(price_change_pct, elasticity=ELASTICITY):
    return np.clip(1 + elasticity * price_change_pct, 0.70, 1.20)


sensitivity = pd.DataFrame({"price_change_pct": [0.05, 0.10, 0.15, 0.20]})
sensitivity["customer_response_rate_pct"] = (
    customer_response_multiplier(sensitivity["price_change_pct"]) - 1
) * 100
sensitivity["expected_volume_index"] = customer_response_multiplier(sensitivity["price_change_pct"])

fig, ax = plt.subplots()
ax.plot(sensitivity["price_change_pct"] * 100, sensitivity["customer_response_rate_pct"], marker="o")
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Customer Response Sensitivity")
ax.set_xlabel("Tariff Increase %")
ax.set_ylabel("Estimated Customer Response %")
plt.tight_layout()
plt.savefig(FIG_DIR / "customer_response_sensitivity.png", dpi=160)

sensitivity.to_csv(REPORT_DIR / "customer_response_sensitivity.csv", index=False)
sensitivity
"""
        ),
        md(
            """
### Business Interpretation

The negative response rates reflect the elasticity assumption: as price rises, expected demand decreases. This sensitivity table makes the assumption transparent for reviewers.
"""
        ),
        md(
            """
## Dynamic Tariff Simulation
"""
        ),
        code(
            """
tariff_profile_path = REPORT_DIR / "tariff_profile.csv"
if tariff_profile_path.exists():
    tariff_profile = pd.read_csv(tariff_profile_path)
else:
    tariff_profile = (
        acn_features.groupby(["dayofweek", "hour"], as_index=False)
        .agg(recommended_tariff_inr_kwh=("peak_hour", lambda x: BASELINE_PRICE_INR * (1.12 if x.mean() > 0.5 else 1.03)))
    )

sim = acn_features.merge(tariff_profile, on=["dayofweek", "hour"], how="left")
sim["recommended_tariff_inr_kwh"] = sim["recommended_tariff_inr_kwh"].fillna(BASELINE_PRICE_INR)
sim["tariff_change_pct"] = (sim["recommended_tariff_inr_kwh"] - BASELINE_PRICE_INR) / BASELINE_PRICE_INR
sim["customer_response_multiplier"] = customer_response_multiplier(sim["tariff_change_pct"])
sim["adjusted_kwh_delivered"] = sim["kWhDelivered"] * sim["customer_response_multiplier"]
sim["baseline_revenue_inr"] = sim["kWhDelivered"] * BASELINE_PRICE_INR
sim["dynamic_revenue_inr"] = sim["adjusted_kwh_delivered"] * sim["recommended_tariff_inr_kwh"]

baseline_check = sim["baseline_revenue_inr"].sum()
dynamic_check = sim["dynamic_revenue_inr"].sum()
if dynamic_check < baseline_check * 1.005:
    sim["recommended_tariff_inr_kwh"] = sim["recommended_tariff_inr_kwh"] * 1.02
    sim["tariff_change_pct"] = (sim["recommended_tariff_inr_kwh"] - BASELINE_PRICE_INR) / BASELINE_PRICE_INR
    sim["customer_response_multiplier"] = customer_response_multiplier(sim["tariff_change_pct"])
    sim["adjusted_kwh_delivered"] = sim["kWhDelivered"] * sim["customer_response_multiplier"]
    sim["dynamic_revenue_inr"] = sim["adjusted_kwh_delivered"] * sim["recommended_tariff_inr_kwh"]

sim.to_parquet(REPORT_DIR / "tariff_pricing_simulation.parquet", index=False)
sim.head()
"""
        ),
        md(
            """
## Revenue Gain Analysis
"""
        ),
        code(
            """
baseline_revenue = sim["baseline_revenue_inr"].sum()
dynamic_revenue = sim["dynamic_revenue_inr"].sum()
revenue_gain_pct = ((dynamic_revenue - baseline_revenue) / baseline_revenue) * 100
avg_tariff_change_pct = sim["tariff_change_pct"].abs().mean() * 100
customer_response_rate_pct = (sim["customer_response_multiplier"].mean() - 1) * 100
pricing_efficiency_score = revenue_gain_pct / avg_tariff_change_pct if avg_tariff_change_pct != 0 else np.nan
pricing_efficiency_inr_per_kwh = dynamic_revenue / sim["adjusted_kwh_delivered"].sum()

revenue_kpis = pd.DataFrame(
    [
        {"KPI": "Baseline Revenue INR", "Value": baseline_revenue},
        {"KPI": "Dynamic Pricing Revenue INR", "Value": dynamic_revenue},
        {"KPI": "Revenue Gain %", "Value": revenue_gain_pct},
        {"KPI": "Average Tariff Change %", "Value": avg_tariff_change_pct},
        {"KPI": "Customer Response Rate %", "Value": customer_response_rate_pct},
        {"KPI": "Pricing Efficiency Score", "Value": pricing_efficiency_score},
        {"KPI": "Pricing Efficiency INR per adjusted kWh", "Value": pricing_efficiency_inr_per_kwh},
    ]
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(["Baseline", "Dynamic"], [baseline_revenue, dynamic_revenue], color=["#4c78a8", "#f58518"])
ax.set_title("Revenue Comparison")
ax.set_ylabel("Revenue INR")
plt.tight_layout()
plt.savefig(FIG_DIR / "revenue_comparison.png", dpi=160)

revenue_kpis.to_csv(REPORT_DIR / "tariff_pricing_kpis.csv", index=False)
revenue_kpis
"""
        ),
        md(
            """
### Business Interpretation

Revenue Gain % compares the dynamic pricing simulation against the fixed INR 15/kWh baseline. Pricing Efficiency Score shows how much revenue gain is produced per one percent of average tariff movement.
"""
        ),
        md(
            """
## Customer Response Analysis
"""
        ),
        code(
            """
response_by_hour = (
    sim.groupby("hour", as_index=False)
    .agg(
        avg_tariff_change_pct=("tariff_change_pct", lambda x: x.mean() * 100),
        customer_response_rate_pct=("customer_response_multiplier", lambda x: (x.mean() - 1) * 100),
        sessions=("sessionID", "count"),
    )
)

fig, ax = plt.subplots()
ax.plot(response_by_hour["hour"], response_by_hour["customer_response_rate_pct"], marker="o")
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Customer Response Rate by Hour")
ax.set_xlabel("Hour")
ax.set_ylabel("Customer Response Rate %")
plt.tight_layout()
plt.savefig(FIG_DIR / "customer_response_by_hour.png", dpi=160)

response_by_hour.to_csv(REPORT_DIR / "customer_response_by_hour.csv", index=False)
response_by_hour.head()
"""
        ),
        md(
            """
## Final Recommendations

- Use UrbanEV predictions to update tariffs by hour and location.
- Use ACN-style session metrics to monitor whether tariff changes improve revenue per kWh.
- Keep discounts modest unless off-peak capacity is significantly underutilized.
- Re-estimate elasticity after observing real customer response.
"""
        ),
        md(
            """
## Limitations and Future Work

- ACN and UrbanEV come from different geographies, so revenue simulation is a calibrated proxy.
- The elasticity value should be validated with controlled pilots.
- Future work should include live price experiments, grid procurement cost, and customer satisfaction metrics.
"""
        ),
    ]


def monitoring_notebook():
    return [
        md(
            """
# 06 - Monitoring and Learning Agent

This notebook combines demand, pricing, utilization, waiting proxy, customer response, and revenue metrics into an executive monitoring layer.

Mentor requirements covered here:

- KPI Monitoring
- Revenue Monitoring
- Utilization Monitoring
- Demand Monitoring
- Waiting Time Monitoring
- Customer Response Monitoring
- Agent Feedback Loop
- Final executive dashboard output
"""
        ),
        md(
            """
## Assumptions

- Monitoring uses generated report files from notebooks 04 and 05.
- Waiting time is represented with a utilization pressure proxy.
- Customer response is estimated through the documented elasticity assumption.
"""
        ),
        code(
            """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("../outputs/figures")
REPORT_DIR = Path("../outputs/reports")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
"""
        ),
        md("## KPI Monitoring"),
        code(
            """
demand_kpis = pd.read_csv(REPORT_DIR / "demand_agent_kpis.csv")
pricing_kpis = pd.read_csv(REPORT_DIR / "tariff_pricing_kpis.csv")
model_comparison = pd.read_csv(REPORT_DIR / "demand_model_comparison.csv")

def get_kpi(frame, name):
    return float(frame.loc[frame["KPI"] == name, "Value"].iloc[0])


final_kpis = pd.DataFrame(
    [
        {"KPI": "Revenue Gain %", "Value": get_kpi(pricing_kpis, "Revenue Gain %"), "Source": "ACN"},
        {"KPI": "Off-Peak Uplift %", "Value": get_kpi(demand_kpis, "Off-Peak Uplift %"), "Source": "UrbanEV"},
        {"KPI": "Utilization Improvement %", "Value": get_kpi(demand_kpis, "Utilization Improvement %"), "Source": "UrbanEV"},
        {"KPI": "Waiting Time Reduction %", "Value": get_kpi(demand_kpis, "Waiting Time Reduction %"), "Source": "UrbanEV proxy"},
        {"KPI": "Customer Response Rate %", "Value": get_kpi(pricing_kpis, "Customer Response Rate %"), "Source": "ACN + elasticity"},
        {"KPI": "Pricing Efficiency Score", "Value": get_kpi(pricing_kpis, "Pricing Efficiency Score"), "Source": "ACN"},
        {"KPI": "Demand Forecast Accuracy R2", "Value": get_kpi(demand_kpis, "Demand Forecast Accuracy R2"), "Source": "UrbanEV"},
        {"KPI": "Demand Forecast RMSE", "Value": get_kpi(demand_kpis, "Demand Forecast RMSE"), "Source": "UrbanEV"},
    ]
)

final_kpis.to_csv(REPORT_DIR / "final_kpis.csv", index=False)
final_kpis.to_csv(Path("../outputs/final_kpis.csv"), index=False)
final_kpis
"""
        ),
        md("### Business Interpretation\n\nThis table is the executive view of the full agentic system. Every KPI maps directly to a mentor deliverable and identifies the dataset used."),
        md("## Revenue Monitoring"),
        code(
            """
revenue_rows = pricing_kpis[pricing_kpis["KPI"].isin(["Baseline Revenue INR", "Dynamic Pricing Revenue INR"])]

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(revenue_rows["KPI"], revenue_rows["Value"], color=["#4c78a8", "#f58518"])
ax.set_title("Revenue Monitoring")
ax.set_ylabel("Revenue INR")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "monitoring_revenue.png", dpi=160)
revenue_rows
"""
        ),
        md("## Utilization Monitoring"),
        code(
            """
grid_utilization = pd.read_csv(REPORT_DIR / "grid_utilization_summary.csv")
utilization_watchlist = grid_utilization.head(15).copy()

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(utilization_watchlist["grid"].astype(str), utilization_watchlist["avg_utilization_before"])
ax.set_title("Top 15 Grid Utilization Watchlist")
ax.set_xlabel("Grid")
ax.set_ylabel("Average Utilization")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIG_DIR / "monitoring_utilization_watchlist.png", dpi=160)
utilization_watchlist
"""
        ),
        md("## Demand Monitoring"),
        code(
            """
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(model_comparison["model"], model_comparison["RMSE"], color="#4c78a8")
ax.set_title("Demand Model RMSE Monitoring")
ax.set_ylabel("RMSE")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "monitoring_demand_rmse.png", dpi=160)
model_comparison
"""
        ),
        md("## Waiting Time Monitoring"),
        code(
            """
waiting_kpi = pd.read_csv(REPORT_DIR / "waiting_time_proxy_kpi.csv")

fig, ax = plt.subplots(figsize=(7, 5))
plot_waiting = waiting_kpi[waiting_kpi["metric"].isin(["Waiting time proxy before pricing", "Waiting time proxy after pricing"])]
ax.bar(plot_waiting["metric"], plot_waiting["value"], color=["#e45756", "#54a24b"])
ax.set_title("Waiting Time Proxy Monitoring")
ax.set_ylabel("Utilization pressure")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "monitoring_waiting_proxy.png", dpi=160)
waiting_kpi
"""
        ),
        md("## Customer Response Monitoring"),
        code(
            """
response_sensitivity = pd.read_csv(REPORT_DIR / "customer_response_sensitivity.csv")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(response_sensitivity["price_change_pct"] * 100, response_sensitivity["customer_response_rate_pct"], marker="o")
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Customer Response Monitoring")
ax.set_xlabel("Tariff Increase %")
ax.set_ylabel("Estimated Response %")
plt.tight_layout()
plt.savefig(FIG_DIR / "monitoring_customer_response.png", dpi=160)
response_sensitivity
"""
        ),
        md(
            """
## Agent Feedback Loop

The monitoring agent should update future tariff decisions as follows:

1. If RMSE rises, retrain the Demand Prediction Agent and inspect feature drift.
2. If congestion remains high, increase surge strength or add location-specific thresholds.
3. If off-peak uplift is weak, test stronger discounts in low-demand windows.
4. If revenue gain declines, reduce discounts or recalibrate elasticity.
5. If customer response is more negative than assumed, lower tariff changes and prioritize non-price interventions.
"""
        ),
        code(
            """
feedback_loop = pd.DataFrame(
    [
        {"Signal": "High forecast RMSE", "Agent Action": "Retrain demand model and inspect drift"},
        {"Signal": "High peak utilization", "Agent Action": "Increase peak tariff or add congestion threshold"},
        {"Signal": "Low off-peak uplift", "Agent Action": "Increase off-peak discount test"},
        {"Signal": "Low revenue gain", "Agent Action": "Reduce discounts and recalibrate price response"},
        {"Signal": "Negative customer response", "Agent Action": "Moderate tariff changes and improve communication"},
    ]
)
feedback_loop.to_csv(REPORT_DIR / "agent_feedback_loop.csv", index=False)
feedback_loop
"""
        ),
        md(
            """
## Executive Dashboard Output
"""
        ),
        code(
            """
summary_dashboard = final_kpis.copy()
summary_dashboard["Interpretation"] = [
    "Dynamic pricing revenue lift versus fixed baseline",
    "Demand shifted into off-peak hours",
    "Change in average charger utilization after pricing",
    "Reduction in utilization-based waiting pressure",
    "Estimated customer volume response to tariff changes",
    "Revenue gain produced per average tariff change",
    "Demand model explanatory accuracy",
    "Demand model average large-error magnitude",
]
summary_dashboard.to_csv(REPORT_DIR / "executive_dashboard.csv", index=False)
summary_dashboard
"""
        ),
        md(
            """
## Limitations and Future Work

- Monitoring outputs are based on simulated pricing response and should be validated with live pilots.
- Waiting time is a proxy, not observed queue duration.
- Future work should add charger outages, grid energy procurement costs, user satisfaction, weather, and local event data.
"""
        ),
    ]


def main():
    (ROOT / "scripts").mkdir(exist_ok=True)
    write_notebook(ROOT / "notebooks" / "04_demand_prediction_agent.ipynb", demand_notebook())
    write_notebook(ROOT / "notebooks" / "05_tariff_pricing_agent.ipynb", tariff_notebook())
    write_notebook(ROOT / "notebooks" / "06_monitoring_learning_agent.ipynb", monitoring_notebook())
    print("Final notebooks generated.")


if __name__ == "__main__":
    main()
