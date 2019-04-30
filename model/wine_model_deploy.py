import pandas as pd
import pathlib

from environs import Env

import mlflow.azureml

from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice

ENV = Env()
ENV.read_env()

data_path = pathlib.Path().absolute() / "data/wine-quality.csv"

# Create or load an existing Azure ML workspace. You can also load an existing workspace using
# Workspace.get(name="<workspace_name>")
workspace_name = ENV("AML_WORKSPACE")
subscription_id = ENV("AZURE_SUBSCRIPTION_ID")
resource_group = ENV("AZURE_RESOURCE_GROUP")
location = ENV("AZURE_RESOURCE_LOCATION")
azure_workspace = Workspace.create(name=workspace_name,
                                   subscription_id=subscription_id,
                                   resource_group=resource_group,
                                   location=location,
                                   create_resource_group=True,
                                   exist_okay=True)

# Build an Azure ML container image for deployment
azure_image, azure_model = mlflow.azureml.build_image(model_path="<path-to-model>",
                                                      workspace=azure_workspace,
                                                      description="Wine regression model 1",
                                                      synchronous=True)
# If your image build failed, you can access build logs at the following URI:
print("Access the following URI for build logs: {}".format(azure_image.image_build_log_uri))

# Deploy the container image to ACI
webservice_deployment_config = AciWebservice.deploy_configuration()
webservice = Webservice.deploy_from_image(
                    image=azure_image, workspace=azure_workspace, name="<deployment-name>")
webservice.wait_for_deployment()

# After the image deployment completes, requests can be posted via HTTP to the new ACI
# webservice's scoring URI. The following example posts a sample input from the wine dataset
# used in the MLflow ElasticNet example:
# https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine
print("Scoring URI is: %s", webservice.scoring_uri)

import requests
import json

# `sample_input` is a JSON-serialized pandas DataFrame with the `split` orientation
sample_input = pd.read_csv(data_path).head().drop(["quality"], axis=1).to_json(orient='split')

response = requests.post(
              url=webservice.scoring_uri, data=json.dumps(sample_input),
              headers={"Content-type": "application/json"})
response_json = json.loads(response.text)
print(response_json)