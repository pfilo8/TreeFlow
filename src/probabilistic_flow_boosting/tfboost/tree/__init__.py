from .ecatboost import EmbeddableCatBoostPriorNormal, EmbeddableCatBoostPriorPredicted, EmbeddableCatBoostPriorAveraged
from .engboost import EmbeddableNGBoost
from .eonehotencoder import EmbeddableOneHotEncoder

MODELS = {
    "CatBoostPriorNormal": EmbeddableCatBoostPriorNormal,
    "CatBoostPriorPredicted": EmbeddableCatBoostPriorPredicted,
    "CatBoostPriorAveraged": EmbeddableCatBoostPriorAveraged
}
