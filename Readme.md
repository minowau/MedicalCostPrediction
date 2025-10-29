# 🏥 Medical Cost Prediction – Exploratory Data Analysis (EDA) & Modeling

## 📘 Project Overview
This project aims to **analyze and predict medical or transport costs** using machine learning techniques. 
The notebook covers **data loading, exploratory data analysis, preprocessing, feature engineering, and multiple regression model training** 
to estimate cost outcomes from patient or transport-related features.

---

## 📂 Dataset
- **train.csv** – Dataset used for model training and exploration.  
- **test.csv** – Dataset for generating predictions.  


Each record represents aggregated medical or transport instances with various numerical and categorical features that influence overall cost.

---

## 🧩 Objectives
- Perform an in-depth **Exploratory Data Analysis (EDA)** to uncover patterns and relationships.
- Identify missing values, data imbalance, and distribution skewness.
- Engineer meaningful features for predictive modeling.
- Compare multiple regression algorithms to find the best-performing model.
- Generate reproducible and submission-ready predictions.

---

## 🧠 EDA Highlights
The notebook performs detailed analysis and visualization steps, including:
- Data inspection using `.info()`, `.describe()`, and null-value checks.
- Univariate analysis using histograms, boxplots, and KDE plots.
- Bivariate analysis between numerical and categorical variables.
- Correlation heatmaps to identify redundant or related variables.
- Outlier detection using **IQR** and **Z-score** techniques.
- Data transformations (log, power) for skewed features.

---

## ⚙️ Preprocessing & Feature Engineering
Key preprocessing techniques implemented:
- **Imputation**: Missing values handled via `SimpleImputer`.
- **Encoding**: Categorical variables encoded using `OneHotEncoder`.
- **Scaling**: Numerical features scaled with `RobustScaler` and transformed with `PowerTransformer`.
- **Polynomial Features**: Created interaction and quadratic terms to capture nonlinear relationships.
- **Feature Selection**: Filtered low-variance and redundant features.
- **Target Transformation**: Log-transform applied for stabilizing variance and improving model performance.

---

## 🤖 Machine Learning Models Used
The project implements multiple regression models via scikit-learn pipelines:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **ElasticNet**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- **LightGBM Regressor**
- **K-Nearest Neighbors (KNN) Regressor**
- **Stacking Regressor** combining multiple base learners

Model selection and performance validation are done using:
- **Cross-validation (KFold)** with `N_SPLITS_CV = 3`
- **Metrics:** RMSE, MAE, and R²

---

## 📈 Results & Observations
- Ridge and XGBoost models demonstrated strong generalization performance.
- Feature scaling and power transformation significantly reduced RMSE.
- Feature importance visualizations highlight key predictors influencing cost.
- Stacking ensemble achieved marginal improvement over individual models.

---

## 📊 Visualization Examples
| Plot Type | Purpose |
|------------|----------|
| Distribution Plots | Visualize skewness and outliers |
| Correlation Heatmap | Identify multicollinearity |
| Boxplots | Compare feature distributions by category |
| Scatterplots | Study numerical feature interactions |
| SHAP / Feature Importance | Interpret model decisions |

---

## 🧰 Libraries Used
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scipy.stats`
- `scikit-learn`
- `xgboost`, `lightgbm`

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python -m notebook
# Open MedicalcostPrediction.ipynb
```
Then execute all cells in order to reproduce preprocessing, training, and submission steps.

