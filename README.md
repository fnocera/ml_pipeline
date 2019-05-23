# ml_pipeline
Exploration of different ML Pipeline options.

# Setup

My setup is usually always the same and pretty straightforward: 
```
python3 -m venv venv  # This creates a venv called "venv"
source venv/bin/activate  # Activate the venv
pip install -r requirements.txt  # Install dependencies
ipython kernel install --name=wine_env --display-name="wine_env" --user  # Create an ipykernel link to venv so I can use it in jupyter notebook
jupyter notebook  # Launch jupyter notebook!

```


You can develop your model with training pipeline and run with whatever variables you would like. 
You can then package it as an MLProject and run with this command (replacing variables with whatever you prefer using). 
```
mlflow run model/wine_model_train.py -P model_name="model_1" alpha=0.8 l1_ratio=0.4
```

TODO: Add details of ML Flow and AML details.

To deploy a server using an existing Azure Blob storage run: 
```
mlflow server --file-store /tmp/mlflow --default-artifact-root "$MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT" --host 0.0.0.0
``` 
Then you can link to the server/where it is hosted for the deployment part. `MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT=wasbs://container@blob)NAME.blob.core.windows.net

