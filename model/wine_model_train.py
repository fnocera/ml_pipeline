#!/usr/bin/env python
# coding: utf-8

# ## Example from ML Flow website

import pandas as pd
import pathlib
import numpy as np
import sys
from environs import Env

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

np.random.seed(20)

ENV = Env()
ENV.read_env()

# To set up the server to store on blob I had to run: Make sure you pip install azure-storage
# mlflow server --file-store /tmp/mlflow --default-artifact-root "$MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT" --host 0.0.0.0
# Then run: python model/wine_model_train.py "model_test_wine" 0.7 0.2



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(model_name, alpha, l1_ratio):

    data_path = pathlib.Path().absolute() / "data/wine-quality.csv"
    data = pd.read_csv(data_path)

    # Split the data into training and test sets.
    train, test = train_test_split(data, test_size=0.3)

    # The predicted column is "quality". 
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # This should reflect wherever you have the server running and attach whatever storage you want.
    mlflow.tracking.set_tracking_uri("http://0.0.0.0:5000")

    # Set experiment which will create experiment if it does not already exist
    experiment_id = mlflow.set_experiment("wine_exp")

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run(experiment_id=experiment_id):
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("RMSE: %s" % rmse)
        print("MAE: %s" % mae)
        print("R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, model_name)

# Currently just storing locally but ideally want to do it in Azure blob:
# wasbs://<container>@<storage-account>.blob.core.windows.net/<path>
# set AZURE_STORAGE_CONNECTION_STRING

"""
mlflow server 
--backend-store-uri wasbs://mlflowserver@fenoceraamlwor0346003249.blob.core.windows.net/runs
--default-artifact-root wasbs://mlflowserver@fenoceraamlwor0346003249.blob.core.windows.net/artifacts

"""


if __name__ == "__main__":
    model_name = sys.argv[1]
    alpha = float(sys.argv[2])
    l1 = float(sys.argv[3])
    train(model_name, alpha, l1)
