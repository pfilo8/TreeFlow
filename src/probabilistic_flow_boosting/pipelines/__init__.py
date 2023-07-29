from kedro.pipeline import pipeline

from .modeling import create_pipeline_train_model, create_pipeline_train_model_ngboost
from .reporting import create_pipeline_report, create_pipeline_report_ngboost

from .reporting import create_pipeline_aggregated_report


def create_general_pipeline(namespace):
    pipeline_general = create_pipeline_train_model() + create_pipeline_report()

    p = pipeline(
        pipeline_general,
        namespace=namespace
    )
    return p


def create_general_pipeline_ngboost(namespace):
    pipeline_general = create_pipeline_train_model_ngboost() + create_pipeline_report_ngboost()

    p = pipeline(
        pipeline_general,
        namespace=namespace
    )
    return p
