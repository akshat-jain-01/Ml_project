##End to end Datascience project

import dagshub
dagshub.init(repo_owner='akshatjainkht01', repo_name='Ml_project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)