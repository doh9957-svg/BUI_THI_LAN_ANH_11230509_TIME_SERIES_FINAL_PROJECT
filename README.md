# Time Series Forecasting Final Project

Final Project for the **Time Series** course.

This project implements a complete end-to-end **multivariate time series forecasting pipeline** using:

- Classical Statistical Models
- Machine Learning
- Deep Learning

The project focuses on forecasting climate variables from the **Daily Delhi Climate Dataset**, with the primary target variable:

```text
meantemp
```

---

## System Features

The system includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Walk-forward validation
- Forecast visualization
- Model evaluation
- Model persistence/export

---

## Student Information

| Information | Value |
|---|---|
| Student ID | 11230509 |
| Course | Time Series |
| Project Type | Final Project |

---

## Project Structure

```text
FINAL_PROJECT/
│
├── data/
│   ├── DailyDelhiClimateTrain.csv
│   └── DailyDelhiClimateTest.csv
│
├── notebook/
│   └── pipeline_full.ipynb
│
├── outputs/
│   ├── adf_test.csv
│   ├── correlation_matrix.csv
│   ├── data_types.csv
│   ├── feature_target_correlation.csv
│   ├── forecast_predictions.csv
│   ├── leaderboard.csv
│   ├── missing_values.csv
│   ├── test_summary.csv
│   └── train_summary.csv
│
├── plots/
│   └── Generated visualization plots
│
├── saved_models/
│   ├── GRU.h5
│   ├── LSTM.h5
│   ├── SimpleRNN.h5
│   └── XGBoost.pkl
│
├── src/
│   ├── config.py
│   ├── dl_models.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── ml_models.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── refactored_pipeline.py
│   ├── statistical_models.py
│   ├── utils.py
│   └── visualization.py
│
├── venv/
│
├── requirements.txt
│
└── README.md
```

---

## Dataset

Dataset used:

```text
Daily Delhi Climate Dataset
```

### Features

| Feature | Description |
|---|---|
| meantemp | Mean temperature |
| humidity | Humidity level |
| wind_speed | Wind speed |
| meanpressure | Atmospheric pressure |

### Dataset Split

- Training set
- Testing set

---

## Project Objectives

The main objectives of this project are:

- Perform comprehensive EDA on multivariate time series data
- Apply feature engineering for forecasting improvement
- Implement walk-forward validation
- Compare multiple forecasting approaches
- Evaluate model performance using regression metrics
- Save trained models and prediction outputs

---

# Models Implemented

## 1. Classical Statistical Models

### ARIMA

AutoRegressive Integrated Moving Average model.

### SARIMA

Seasonal ARIMA model with seasonality handling.

### VAR

Vector AutoRegression for multivariate forecasting.

---

## 2. Machine Learning

### XGBoost Regressor

Gradient boosting model for time series forecasting.

### Features Used

- Lag features
- Rolling statistics
- Calendar features

---

## 3. Deep Learning Models

### SimpleRNN

Basic recurrent neural network.

### LSTM

Long Short-Term Memory network for sequential learning.

### GRU

Gated Recurrent Unit network.

---

# Pipeline Workflow

The forecasting pipeline consists of the following stages:

1. Data Loading
2. Data Cleaning
3. Outlier Handling
4. Exploratory Data Analysis
5. Feature Engineering
6. Data Scaling
7. Sequence Generation
8. Model Training
9. Walk-Forward Validation
10. Model Evaluation
11. Forecast Visualization
12. Model Saving
13. Report Export

---

# Exploratory Data Analysis (EDA)

The project includes comprehensive EDA.

## Summary Statistics

- Train/Test summaries
- Data types
- Missing values analysis

## Visualization

- Time series visualization
- Distribution plots
- Boxplots
- Correlation heatmaps
- Rolling statistics
- Monthly seasonality

## Statistical Analysis

- ACF/PACF
- Lag plots
- Seasonal decomposition
- ADF stationarity test

---

# Feature Engineering

Engineered features include:

## Calendar Features

```text
year
month
day
dayofweek
quarter
```

## Lag Features

```text
meantemp_lag_1
meantemp_lag_2
meantemp_lag_7
```

## Rolling Features

```text
meantemp_roll_mean_3
meantemp_roll_mean_7
meantemp_roll_std_7
```

---

# Walk-Forward Validation

This project uses:

```text
Walk-Forward Validation
```

instead of random train-test splitting.

## Advantages

- Preserves temporal order
- Avoids data leakage
- More realistic forecasting evaluation
- Suitable for production forecasting systems

---

# Evaluation Metrics

The following metrics are used:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

Results are automatically exported to:

```text
outputs/leaderboard.csv
```

---

# Generated Outputs

## Reports

```text
outputs/
```

Contains:

- leaderboard.csv
- forecast_predictions.csv
- train_summary.csv
- test_summary.csv
- missing_values.csv
- correlation_matrix.csv
- adf_test.csv

---

## Saved Models

```text
saved_models/
```

Contains:

- XGBoost.pkl
- SimpleRNN.h5
- LSTM.h5
- GRU.h5

---

## Visualization Plots

```text
plots/
```

Contains:

- Time series plots
- Distribution plots
- Correlation heatmaps
- Seasonal decomposition
- ACF/PACF plots
- Forecast comparison plots
- Residual analysis plots

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd FINAL_PROJECT
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### macOS/Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Project

## Run Full Pipeline

From project root:

```bash
python src/refactored_pipeline.py
```

or:

```bash
python -m src.refactored_pipeline
```

---

## Jupyter Notebook

Notebook version available:

```text
notebook/pipeline_full.ipynb
```

Run with:

```bash
jupyter notebook
```

---

# Example Pipeline Output

The pipeline automatically:

- Trains all forecasting models
- Evaluates performance
- Saves leaderboard results
- Generates forecast plots
- Exports prediction files
- Saves trained models

---

# Forecasting Models Comparison

| Category | Models |
|---|---|
| Statistical | ARIMA, SARIMA, VAR |
| Machine Learning | XGBoost |
| Deep Learning | SimpleRNN, LSTM, GRU |

---

# Technologies Used

## Programming Language

```text
Python 3.11
```

---

## Libraries

### Data Processing

- pandas
- numpy

### Visualization

- matplotlib
- seaborn

### Statistical Modeling

- statsmodels

### Machine Learning

- scikit-learn
- xgboost

### Deep Learning

- tensorflow
- keras

---

# Key Learning Outcomes

This project demonstrates understanding of:

- Time Series Forecasting
- Statistical Modeling
- Deep Learning for Sequential Data
- Walk-Forward Validation
- Time Series Feature Engineering
- Model Evaluation
- Forecast Visualization
- End-to-End ML Pipeline Design

---

# Future Improvements

Potential future enhancements:

- Hyperparameter optimization
- Prophet model integration
- Transformer-based forecasting
- Multistep forecasting
- Advanced ensemble forecasting
- Attention-based deep learning architectures
- Automated hyperparameter tuning

---

# Author

| Information | Value |
|---|---|
| Student ID | 11230509 |
| Course | Time Series |
| Project | Final Project |

---

# License

This project is for academic and educational purposes only.
