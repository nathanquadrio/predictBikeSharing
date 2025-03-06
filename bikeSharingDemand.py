import pandas as pd                                     # type: ignore
import numpy as np                                      # type: ignore
from sklearn.model_selection import train_test_split    # type: ignore
import lightgbm as lgb                                  # type: ignore    
import xgboost as xgb                                   # type: ignore
from catboost import CatBoostRegressor                  # type: ignore
from skopt import BayesSearchCV                         # type: ignore
from sklearn.ensemble import StackingRegressor          # type: ignore
from sklearn.linear_model import Ridge                  # type: ignore

# Code used to complete step 4 and achieve the Kaggle score of 0.40352

# Load data
train = pd.read_csv("~/Documents/AWS_MLEngineering/project/input/train.csv", parse_dates=['datetime'])
test = pd.read_csv("~/Documents/AWS_MLEngineering/project/input/test.csv", parse_dates=['datetime'])
submission = pd.read_csv("~/Documents/AWS_MLEngineering/project/input/sampleSubmission.csv", parse_dates=['datetime'])
print("data loaded...")

# Feature engineering
for df in [train, test]:
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["year"] = df["datetime"].dt.year
    df["season"] = df["season"].astype("category")
    df["weather"] = df["weather"].astype("category")

print("feature engineering done...")

# Prepare the data
X = train.drop(columns=["datetime", "count", "casual", "registered"])
y = np.log1p(train["count"])  # Log-transform target
X_test = test.drop(columns=["datetime"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("data prepared...")

# Train the model
lgb_search = BayesSearchCV(
    lgb.LGBMRegressor(),
    {
        "n_estimators": (200, 1000),
        "learning_rate": (0.01, 0.3),
        "num_leaves": (20, 200),
        "max_depth": (3, 15),
        "min_child_samples": (10, 100),
    },
    n_iter=50, cv=3, n_jobs=-1
)
lgb_search.fit(X_train, y_train)
best_lgb = lgb_search.best_estimator_

print("LGB trained...")

xgb_search = BayesSearchCV(
    xgb.XGBRegressor(enable_categorical=True),
    {
        "n_estimators": (200, 1000),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 15),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
    },
    n_iter=50, cv=3, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

print("XGB trained...")

# Train CatBoost
cat_features = ["season", "weather"]
cat_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, verbose=0)
cat_model.fit(X_train, y_train, cat_features=cat_features)

print("CatBoost trained...")

# Convert categorical features to integer codes
for df in [X_train, X_val, X_test]:
    df["season"] = df["season"].cat.codes
    df["weather"] = df["weather"].cat.codes

# Stacking Models
stacking_model = StackingRegressor(
    estimators=[("lgb", best_lgb), ("xgb", best_xgb), ("cat", cat_model)],
    final_estimator=Ridge(alpha=1.0),
    passthrough=True
)
stacking_model.fit(X_train, y_train)

predictions = np.expm1(stacking_model.predict(X_test))  # Convert back from log scale
predictions = np.clip(predictions, 0, None)             # Ensure non-negative values

submission = pd.DataFrame({"datetime": test["datetime"], "count": predictions})
submission.to_csv("~/Documents/AWS_MLEngineering/project/output/submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
