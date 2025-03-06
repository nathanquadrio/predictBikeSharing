import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("~/Documents/AWS_MLEngineering/project/input/train.csv", parse_dates=['datetime'])
test = pd.read_csv("~/Documents/AWS_MLEngineering/project/input/test.csv", parse_dates=['datetime'])
submission = pd.read_csv("~/Documents/AWS_MLEngineering/project/input/sampleSubmission.csv", parse_dates=['datetime'])

# Feature Engineering: Extract DateTime Components
for df in [train, test]:
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["year"] = df["datetime"].dt.year
    df["season"] = df["season"].astype("category")
    df["weather"] = df["weather"].astype("category")
	
## Prepare Data
# Drop unnecessary columns
X = train.drop(columns=["datetime", "count", "casual", "registered"])
y = train["count"]
X_test = test.drop(columns=["datetime"])

# Split data into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
print("Training LightGBM...")
lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03)
lgb_model.fit(X_train, y_train)

print("Training CatBoost...")
cat_features = ["season", "weather"]
cat_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, verbose=0)
cat_model.fit(X_train, y_train, cat_features=cat_features)

print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, enable_categorical=True)
xgb_model.fit(X_train, y_train)

# ================================
print("Generating predictions...")
pred_lgb = lgb_model.predict(X_test)
pred_cat = cat_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)

print("Creating final weighted ensemble...")
final_predictions = (0.4 * pred_lgb) + (0.3 * pred_cat) + (0.3 * pred_xgb)

# Ensure predictions are non-negative
final_predictions = np.clip(final_predictions, 0, None)

submission = pd.DataFrame({"datetime": test["datetime"], "count": final_predictions})
submission.to_csv("~/Documents/AWS_MLEngineering/project/output/submission.csv", index=False)
