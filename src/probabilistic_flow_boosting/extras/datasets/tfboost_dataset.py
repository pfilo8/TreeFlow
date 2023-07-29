from typing import Any, Dict

from kedro.io import AbstractDataSet
from ...tfboost.tfboost import TreeFlowBoost


class TFBoostDataSet(AbstractDataSet):

    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self) -> TreeFlowBoost:
        return TreeFlowBoost.load(self._filepath)

    def _save(self, model: TreeFlowBoost) -> None:
        return model.save(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(
            filepath=self._filepath
        )
