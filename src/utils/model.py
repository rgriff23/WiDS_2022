import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from sklearn.model_selection import GridSearchCV

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def _generate_one_year_holdout_splits(df: pd.DataFrame, holdout_years: list):
    """
    For each holdout year, the training data includes all the data outside
    that year and the test data includes all the data in the holdout year.
    """
    splits = list()

    print(f"Generating {len(holdout_years)} splits.\n")

    for split_ix, year in enumerate(holdout_years):
        train_ix = df[df['Year_Factor'] != year].index.tolist()
        test_ix = df[df['Year_Factor'] == year].index.tolist()
        splits.append((train_ix, test_ix))

        logging.info(f"Cross-validation strategy: \n  3-fold, split using holdout years {holdout_years}.")
        logging.info(f"    Split {split_ix + 1}:\n")
        logging.info(f"      - Training rows: {len(train_ix)}")
        logging.info(f"      - Training years: {df['Year_Factor'].unique().tolist()}")
        logging.info(f"      - Test rows: {len(test_ix)}")
        logging.info(f"      - Test years: {year}")

    return splits


def grid_search_optimization(train: pd.DataFrame, config: dict):
    """

    :param train: Training data
    :param config: Pipeline configuration
    :return: Grid search object, including the final fitted model
    """
    logging.info("********************************")
    logging.info("*** GRID SEARCH OPTIMIZATION ***")
    logging.info("********************************\n")

    full_pipeline = Pipeline(
        steps=[("preprocessor", ColumnTransformer(
            transformers=[
                ("num", config['numeric_transformer'], config['numeric_features']),
                ("wea", config['weather_transformer'], config['weather_features']),
                ("cat", config['categorical_transformer'], config['categorical_features']),
            ])),
               #("feature_selector", config['feature_selector']),
               ("regressor", config['estimator'])])

    logging.info(f"Optimizing pipeline for {config['scoring']}: \n{full_pipeline}")
    logging.info(f"Hyper-parameter grid: \n{config['param_grid']}")

    grid_search = GridSearchCV(estimator=full_pipeline,
                               param_grid=config['param_grid'],
                               cv=_generate_one_year_holdout_splits(train, config["holdout_splits"]),
                               #cv=GroupKFold(n_splits=6).split(train, train[config['target']], train['Year_Factor']),
                               scoring=config["scoring"],
                               n_jobs=-1,  # Use all processors
                               verbose=2,
                               )

    grid_search.fit(train.drop(config['target'], axis=1),
                    train[config['target']].values.ravel())
    logging.info(f"Best cross-validated RMSE: {round(-grid_search.best_score_, 3)}")
    logging.info(f"Best params:\n {pd.DataFrame.from_records(grid_search.best_params_, index=[0]).T}\n")
    try:
        logging.info(f"XGBoost best ntree: {grid_search.best_estimator_.best_ntree_limit}")
    except:
        pass

    return grid_search
