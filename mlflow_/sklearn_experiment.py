# https://data-campus.ai/announcements/of5ikran8qwzlvww
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

import mlflow
import mlflow.sklearn

dataset = fetch_california_housing(as_frame=True)
df = dataset['frame']

print(df.head())

target_col = 'MedHouseVal'

X, y = df.drop(columns=[target_col]), df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def train(n_estimators, max_depth):
    model = RandomForestRegressor(
        n_estimators = n_estimators,
        max_depth = max_depth,
        criterion = 'squared_error',  # 'mse'
        random_state = 0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = MSE(y_pred, y_test)
    R2 = r2_score(y_pred, y_test)

    return model, mse, R2

with mlflow.start_run():
    cand_n_estimators = [10, 100, 1000]
    cand_max_depth = [1, 5, 10]

    trial = 0
    for n_estimators in cand_n_estimators:
        for max_depth in cand_max_depth:
            with mlflow.start_run(nested=True):
                trial += 1
                model, mse, R2 = train(n_estimators, max_depth)
                print(f"trial {trial}: n_estimators={n_estimators}, max_depth={max_depth}, MSE = {mse:.3}, R2 = {R2:.3}")

                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("R2", R2)

                mlflow.sklearn.log_model(model, "model")
