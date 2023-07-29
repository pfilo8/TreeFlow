import timeit

import catboost
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from src.probabilistic_flow_boosting.pipelines.modeling.utils import setup_random_seed
from src.probabilistic_flow_boosting.pipelines.reporting.nodes import (
    calculate_nll,
    calculate_nll_catboost,
    calculate_crps_catboost,
    calculate_crps_treeflow,
    calculate_rmse,
    calculate_rmse_at_1,
    calculate_rmse_at_2,
    calculate_rmse_catboost,
    calculate_rmse_pgbm
)
from src.probabilistic_flow_boosting.cnf import ContinuousNormalizingFlowRegressor
from src.probabilistic_flow_boosting.pgbm import PGBM
from src.probabilistic_flow_boosting.tfboost.tree import EmbeddableCatBoostPriorNormal
from src.probabilistic_flow_boosting.tfboost.tfboost import TreeFlowBoost
from src.probabilistic_flow_boosting.tfboost.flow import ContinuousNormalizingFlow
from src.probabilistic_flow_boosting.tfboost.tfboost_ablation import TreeFlowWithoutShallow


def get_dataset(dataset, ohe=False):
    if dataset == 'avocado':
        df = pd.read_csv('data/01_raw/CatData/avocado/avocado.csv', index_col=0)
        x = df.drop(columns=['Date', 'AveragePrice'])
        if ohe:
            x = pd.get_dummies(x)
        y = df[['AveragePrice']]
        cat_features_s = ['type', 'year', 'region']
        cat_features_n = [8, 9, 10]
    elif dataset == 'bigmart':
        df = pd.read_csv('data/01_raw/CatData/bigmart/bigmart.csv')
        df['Outlet_Size'] = df['Outlet_Size'].fillna('')
        df['Item_Weight'] = df['Item_Weight'].fillna(0.0)
        x = df.drop(columns=['Item_Identifier', 'Item_Outlet_Sales'])
        if ohe:
            x = pd.get_dummies(x)
        y = np.log10(df[['Item_Outlet_Sales']])
        cat_features_s = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',
                          'Outlet_Type']
        cat_features_n = [1, 3, 5, 7, 8, 9]
    elif dataset == 'diamonds':
        df = pd.read_csv('data/01_raw/CatData/diamonds/diamonds.csv', index_col=0)
        x = df.drop(columns=['price'])
        if ohe:
            x = pd.get_dummies(x)
        y = np.log10(df[['price']])
        cat_features_s = ['cut', 'color', 'clarity']
        cat_features_n = [1, 2, 3]
    elif dataset == 'diamonds2':
        df = pd.read_csv('data/01_raw/CatData/diamonds2/diamonds_dataset.csv')
        x = df.drop(columns=['id', 'url', 'price', 'date_fetched'])
        if ohe:
            x = pd.get_dummies(x)
        y = np.log10(df[['price']])
        cat_features_s = ['shape', 'cut', 'color', 'clarity', 'report', 'type']
        cat_features_n = [0, 2, 3, 4, 5, 6]
    elif dataset == 'laptop':
        df = pd.read_csv('data/01_raw/CatData/laptop/laptop_price.csv', index_col=0, engine='pyarrow')
        df['Weight'] = pd.to_numeric(df['Weight'].str.replace('kg', ''))
        df['Ram'] = pd.to_numeric(df['Ram'].str.replace('GB', ''))
        x = df.drop(columns=['Product', 'Price_euros'])
        if ohe:
            x = pd.get_dummies(x)
        y = np.log10(df[['Price_euros']])
        cat_features_s = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
        cat_features_n = [0, 1, 3, 4, 6, 7, 8]
    elif dataset == 'pakwheels':
        df = pd.read_csv('data/01_raw/CatData/pak-wheels/PakWheelsDataSet.csv', index_col=0)
        df['Engine Capacity(CC)'] = df['Engine Capacity(CC)'].fillna(0.0)
        x = df.drop(columns=['Name', 'Price'])
        if ohe:
            x = pd.get_dummies(x)
        y = np.log10(df[['Price']])
        cat_features_s = ['Make', 'Transmission', 'Engine Type', 'City']
        cat_features_n = [0, 1, 2, 5]
    elif dataset == 'sydney_house':
        df = pd.read_csv('data/01_raw/CatData/sydney_house/SydneyHousePrices.csv')
        df['bed'] = df['bed'].fillna(0.0)
        df['car'] = df['car'].fillna(0.0)
        x = df.drop(columns=['Date', 'Id', 'sellPrice'])
        if ohe:
            x = pd.get_dummies(x)
        y = np.log10(df[['sellPrice']])
        cat_features_s = ['suburb', 'postalCode', 'propType']
        cat_features_n = [0, 1, 5]
    elif dataset == 'wine_reviews':
        df = pd.read_csv('data/01_raw/CatData/wine_reviews/winemag-data_first150k.csv', index_col=0)
        df['country'] = df['country'].fillna('')
        df['province'] = df['province'].fillna('')
        df = df.dropna(subset=['price'])
        x = df.drop(columns=['description', 'price', 'designation', 'region_1', 'region_2', 'winery'])
        if ohe:
            x = pd.get_dummies(x)
        y = df[['price']]
        cat_features_s = ['country', 'province', 'variety']
        cat_features_n = [0, 2, 3]
    else:
        raise ValueError(f'Invalid dataset {dataset}')
    return x, y, cat_features_s, cat_features_n


def modeling(
        model, x_tr, y_tr, x_val, y_val, x_test, y_test, cat_features_s, cat_features_n, random_state=42,
        filename='model'
):
    if model == 'pgbm':
        x_tr = torch.Tensor(x_tr.values)
        x_val = torch.Tensor(x_val.values)
        x_test = torch.Tensor(x_test.values)

        y_tr = torch.Tensor(y_tr.values)
        y_val = torch.Tensor(y_val.values)
        y_test = torch.Tensor(y_test.values)

        print(x_train.shape, x_test.shape)

        def mseloss_objective(yhat, y, sample_weight=None):
            gradient = (yhat - y)
            hessian = torch.ones_like(yhat)
            return gradient, hessian

        def rmseloss_metric(yhat, y, sample_weight=None):
            loss = torch.sqrt(torch.mean(torch.square(yhat - y)))
            return loss

        model = PGBM()

        params = {
            'min_split_gain': 0,
            'min_data_in_leaf': 2,
            'max_leaves': 8,
            'max_bin': 64,
            'learning_rate': 0.1,
            'verbose': 2,
            'early_stopping_rounds': 200,
            'feature_fraction': 1,  # For Pakwheels and Sydney Housing we used 0.2,
            'bagging_fraction': 1,
            'seed': random_state,
            'reg_lambda': 1,
            'device': 'gpu',
            'gpu_device_id': 0,
            'derivatives': 'exact',
            'distribution': 'normal',
            'n_estimators': 2000
        }

        start = timeit.default_timer()
        model.train(
            train_set=(x_tr, y_tr),
            objective=mseloss_objective,
            metric=rmseloss_metric,
            valid_set=(x_val, y_val),
            params=params
        )
        train_time = timeit.default_timer() - start

        model.optimize_distribution(
            x_val,
            y_val.reshape(-1),
            distributions=['normal', 'studentt', 'laplace', 'logistic', 'lognormal', 'gamma', 'gumbel', 'weibull',
                           'negativebinomial'])

        y_test_dist = model.predict_dist(x_test, n_forecasts=1000)
        nll = model.nll(x_test, y_test.reshape(-1)).mean().item()
        crps = model.crps_ensemble(y_test_dist, y_test.reshape(-1)).mean().item()
        rmse = calculate_rmse_pgbm(model, x_test, y_test)
        rmse_at_1 = 0.0
        rmse_at_2 = 0.0
    elif model == 'catboost':
        model = catboost.CatBoostRegressor(
            cat_features=cat_features_s,
            loss_function="RMSEWithUncertainty",
            num_trees=2000,
            random_state=random_state,
            verbose=False
        )

        start = timeit.default_timer()
        model.fit(x_tr, y_tr, eval_set=(x_val, y_val))
        train_time = timeit.default_timer() - start

        nll = calculate_nll_catboost(model, x_test, y_test)
        crps = calculate_crps_catboost(model, x_test, y_test)
        rmse = calculate_rmse_catboost(model, x_test, y_test)
        rmse_at_1 = 0.0
        rmse_at_2 = 0.0
    elif model == 'treeflow':
        tree = EmbeddableCatBoostPriorNormal(
            cat_features=cat_features_n,
            loss_function="RMSEWithUncertainty",
            depth=4,
            num_trees=200,
            random_state=random_state,
            verbose=False
        )
        flow = ContinuousNormalizingFlow(input_dim=1, hidden_dims=(16, 16), context_dim=128, num_blocks=2,
                                         conditional=True)

        treeflow = TreeFlowBoost(tree, flow, embedding_size=128)
        start = timeit.default_timer()
        treeflow.fit(x_tr.values, y_tr.values, x_val.values, y_val.values, n_epochs=1, batch_size=4096, verbose=True)
        train_time = timeit.default_timer() - start

        nll = calculate_nll(treeflow, x_test, y_test, batch_size=1024)
        crps = calculate_crps_treeflow(treeflow, x_test, y_test, num_samples=1000, batch_size=1024).mean().item()
        rmse = calculate_rmse(treeflow, x_test, y_test, num_samples=1000, batch_size=1024)
        rmse_at_1 = calculate_rmse_at_1(treeflow, x_test, y_test, num_samples=1000, batch_size=1024)
        rmse_at_2 = calculate_rmse_at_2(treeflow, x_test, y_test, num_samples=1000, batch_size=1024)
    elif model == 'treeflow_ablation':
        tree = EmbeddableCatBoostPriorNormal(
            cat_features=cat_features_n,
            loss_function="RMSEWithUncertainty",
            depth=4,
            num_trees=200,
            random_state=random_state,
            verbose=False
        )
        flow = ContinuousNormalizingFlow(input_dim=1, hidden_dims=(16, 16), context_dim=200 * 4 ** 2, num_blocks=2,
                                         conditional=True)

        treeflow = TreeFlowWithoutShallow(tree, flow, embedding_size=None)
        start = timeit.default_timer()
        treeflow.fit(x_tr.values, y_tr.values, x_val.values, y_val.values, n_epochs=1, batch_size=4096, verbose=True)
        train_time = timeit.default_timer() - start

        nll = calculate_nll(treeflow, x_test, y_test, batch_size=1024)
        crps = calculate_crps_treeflow(treeflow, x_test, y_test, num_samples=1000, batch_size=1024).mean().item()
        rmse = calculate_rmse(treeflow, x_test, y_test, num_samples=1000, batch_size=1024)
        rmse_at_1 = calculate_rmse_at_1(treeflow, x_test, y_test, num_samples=1000, batch_size=1024)
        rmse_at_2 = calculate_rmse_at_2(treeflow, x_test, y_test, num_samples=1000, batch_size=1024)
    elif model == 'cnf':
        cnf = ContinuousNormalizingFlowRegressor(input_dim=x_test.shape[1], output_dim=1, hidden_dims=(16, 16),
                                                 embedding_dim=128, num_blocks=2)
        start = timeit.default_timer()
        cnf.fit(x_tr.values, y_tr.values, x_val.values, y_val.values, n_epochs=1, batch_size=1024, verbose=True,
                max_patience=20)
        train_time = timeit.default_timer() - start

        nll = cnf.nll(x_test.values, y_test.values)
        crps = cnf.crps(x_test.values, y_test.values, num_samples=1000, batch_size=256)
        rmse = cnf.rmse(x_test, y_test, num_samples=1000, batch_size=256)
        rmse_at_1 = calculate_rmse_at_1(cnf, x_test, y_test, num_samples=1000, batch_size=256)
        rmse_at_2 = calculate_rmse_at_2(cnf, x_test, y_test, num_samples=1000, batch_size=256)
    else:
        raise ValueError(f'Invalid model {model}.')

    return crps, nll, rmse, rmse_at_1, rmse_at_2, train_time


DATASETS_LIST = [
    'avocado',
    'bigmart',
    'diamonds',
    'diamonds2',
    'laptop',
    'pakwheels',
    'sydney_house'
]

MODEL_LIST = [
    'pgbm'
    'catboost',
    'treeflow',
    'cnf',
    'treeflow_ablation'
]

results = []

for dataset in DATASETS_LIST:
    for model in MODEL_LIST:
        for i in range(1, 6):
            setup_random_seed(i)
            ohe = (model == 'pgbm' or model == 'cnf')
            x, y, cat_features_s, cat_features_n = get_dataset(dataset, ohe=ohe)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
            x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=i)

            crps, nll, rmse, rmse_at_1, rmse_at_2, train_time = modeling(model, x_tr, y_tr, x_val, y_val, x_test,
                                                                         y_test,
                                                                         cat_features_s, cat_features_n,
                                                                         random_state=i,
                                                                         filename=f'{model}_{dataset}_{i}')

            print(dataset, i, crps, nll, rmse, rmse_at_1, rmse_at_2, train_time)

            results.append([dataset, model, i, crps, nll, rmse, rmse_at_1, rmse_at_2, train_time])

index = 'all'

r = pd.DataFrame(
    results,
    columns=['dataset', 'model', 'index', 'crps', 'nll', 'rmse', 'rmse_at_1', 'rmse_at_2', 'train_time']
)
r.to_csv(f'results_raw_mixed_{index}.csv', index=False)
g = r.groupby(['dataset', 'model'])[['crps', 'nll', 'rmse', 'rmse_at_1', 'rmse_at_2', 'train_time']].agg(
    [np.mean, np.std]
)
g.to_csv(f'results_mixed_{index}.csv')
