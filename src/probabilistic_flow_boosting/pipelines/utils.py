import datetime

import mlflow


def log_dataframe_artifact(dataframe, name):
    path = f'/tmp/{name}-{str(datetime.datetime.now()).replace(" ", "-")}.csv'
    dataframe.to_csv(path, index=False)
    mlflow.log_artifact(path)
