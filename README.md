# Advanced Time Series Forecasting with Uncertainty Quantification

## Project Overview
This project implements a sophisticated deep learning model for **multivariate time series forecasting** with **uncertainty estimation**. The goal is not only high predictive accuracy but also robust prediction intervals to quantify forecast uncertainty.

The workflow includes:
- Generating or acquiring **synthetic multivariate time series data** with **non-stationarity** and **seasonality** (5 features, 5000+ observations).
- Building an **LSTM neural network with Monte Carlo Dropout** to estimate both point forecasts and **90% prediction intervals**.
- Evaluating the model using standard accuracy metrics (**RMSE, MAE**) and uncertainty-specific metrics (**coverage probability**).
- Visualizing forecasted values along with uncertainty bounds.

---

## Data Generation
- **Synthetic dataset** with 5 interdependent features:
  - `feature_1`: Trend + multiple seasonalities + noise
  - `feature_2`: Lagged version of `feature_1` + noise
  - `feature_3`: Random seasonal combination + noise
  - `feature_4`: Sawtooth signal + trend + noise
  - `feature_5`: Combination of other features + noise
- **Total observations:** 5000+
- Dataset exhibits **trend, multiple seasonal cycles, noise, and non-stationarity**, simulating real-world energy or financial data.

---

## Model Architecture
- **Model type:** LSTM Recurrent Neural Network
- **Layers:**
  - LSTM layer with 64 units → Dropout (MC Dropout)
  - LSTM layer with 32 units → Dropout (MC Dropout)
  - Dense output layer (5 features)
- **Sequence length:** 20 time steps
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Uncertainty estimation:** Monte Carlo Dropout during inference (50 simulations per input)

---

## Training
- **Training data:** 80% of dataset
- **Validation:** 10% of training set
- **Epochs:** 10
- **Batch size:** 64
- **Output:** Point forecast + lower/upper bounds for 90% prediction interval

---

## Evaluation Metrics
| Metric | Value |
|--------|-------|
| RMSE (feature_1) | [calculated by code] |
| MAE (feature_1)  | [calculated by code] |
| Coverage Probability (90% CI) | [calculated by code] |

> Coverage probability measures the percentage of true values that fall within the 90% prediction interval.  

---

## Visualization
- Time series plots for each feature
- Forecast vs. true values for feature_1 with **90% prediction interval**
- Helps visualize both accuracy and uncertainty

---

## How to Run
1. Install required libraries:

```bash
pip install -r requirements.txt
