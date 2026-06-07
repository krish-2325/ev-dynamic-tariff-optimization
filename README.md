# EV Dynamic Tariff Optimization using Multi-Agent AI

## Project Overview

This project develops a Multi-Agent AI system for optimizing EV charging tariffs in urban charging networks. The solution combines demand forecasting, dynamic pricing, and continuous monitoring to improve charger utilization, increase operator revenue, reduce waiting time, and encourage off-peak charging behavior.

The system consists of three intelligent agents:

1. Demand Forecasting Agent
2. Dynamic Tariff Pricing Agent
3. Monitoring & Learning Agent

---

## Problem Statement

EV charging stations often face:

- Peak-hour congestion
- Long waiting times
- Uneven charger utilization
- Underutilized infrastructure during off-peak periods
- Static pricing strategies that do not adapt to demand

This project addresses these challenges through AI-driven dynamic pricing.

---

## Solution Architecture

### Agent 1: Demand Forecasting Agent

Responsibilities:

- Process historical charging-session data
- Engineer temporal and utilization features
- Train and compare forecasting models
- Predict future occupancy and utilization

Models Evaluated:

- XGBoost Regressor
- Ridge Regression
- Baseline Lag Model

Selected Model:

- XGBoost Regressor

Performance:

- R² = 0.9849
- RMSE = 3.0766
- MAE = 1.7354

---

### Agent 2: Dynamic Tariff Pricing Agent

Responsibilities:

- Generate hourly tariff recommendations
- Apply utilization-based pricing
- Incorporate customer price elasticity
- Estimate customer response to tariff changes

Pricing Logic:

- Low utilization → Discount pricing
- Medium utilization → Baseline pricing
- High utilization → Surge pricing

Objectives:

- Increase revenue
- Shift demand to off-peak periods
- Reduce congestion
- Improve overall network utilization

---

### Agent 3: Monitoring & Learning Agent

Responsibilities:

- Monitor operational KPIs
- Evaluate pricing effectiveness
- Generate alerts and recommendations
- Simulate a feedback-learning loop
- Optimize tariff parameters over multiple episodes

Monitored KPIs:

- Revenue Gain %
- Utilization Improvement %
- Off-Peak Demand Uplift %
- Waiting Time Reduction %
- Customer Response Rate %
- Pricing Efficiency Score
- Forecast Accuracy Metrics

---

## Final Results

| KPI | Result |
|------|------:|
| Revenue Gain | 11.26% |
| Utilization Improvement | 6.54% |
| Off-Peak Demand Uplift | 2.09% |
| Waiting Time Reduction | 24.78% |
| Customer Response Rate | 5.84% |
| Pricing Efficiency Score | 0.446 |
| Forecast R² | 0.9849 |
| Forecast RMSE | 3.0766 |

---

## Repository Structure

```text
EV-Dynamic-Tariff/
│
├── notebooks/
│   ├── 01_Demand_Forecasting_Agent.ipynb
│   ├── 02_Tariff_Pricing_Agent.ipynb
│   └── 03_Monitoring_Learning_Agent.ipynb
│
├── data/
│
├── outputs/
│   ├── reports/
│   └── figures/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the notebooks sequentially:

### Step 1

Run:

```text
01_Demand_Forecasting_Agent.ipynb
```

Outputs:

- Forecasted occupancy
- Forecasted utilization
- Model evaluation metrics

### Step 2

Run:

```text
02_Tariff_Pricing_Agent.ipynb
```

Outputs:

- Dynamic tariff recommendations
- Revenue estimates
- Customer response analysis

### Step 3

Run:

```text
03_Monitoring_Learning_Agent.ipynb
```

Outputs:

- KPI dashboards
- Alert engine
- Learning recommendations
- Executive summary

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Jupyter Notebook

---

## Business Impact

The proposed AI-driven pricing framework demonstrates how dynamic tariff optimization can:

- Increase charging network revenue
- Improve charger utilization
- Encourage off-peak charging
- Reduce congestion and waiting times
- Support scalable EV infrastructure planning

---

## Future Improvements

- Real-time tariff updates
- Reinforcement Learning based pricing
- Live charger integration
- User-specific pricing recommendations
- Real-time occupancy prediction

---

## Author

Krish Goyal
Civil Engineering
Indian Institute of Technology Roorkee
