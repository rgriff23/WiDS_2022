import numpy as np
import pandas as pd

from src.utils.io import load_train_test, load_local_data, save_local_submission
from src.utils.preprocess import preprocess_train_test
from src.utils.model import grid_search_optimization

#from src.config.random_forest_config import random_forest_config as config
from src.config.xgboost_config import xgb_config as config

if __name__ == '__main__':
    """
    This script trains a pipeline and outputs predictions.
    """

    out_file = 'xgb_umap.csv'
    reuse_preprocessed_data = False

    # Load train/test data
    train_df, test_df = load_train_test()

    # Preprocess train/test data
    if reuse_preprocessed_data:
        train_df = load_local_data('preprocessed_train.csv')
        test_df = load_local_data('preprocessed_test.csv')
    else:
        train_df, test_df = preprocess_train_test(train_df, test_df, config)

    # Train pipeline using grid search to optimize hyper-parameters
    grid_search = grid_search_optimization(train_df, config)

    # Predict target using best estimator from grid search
    predictions_df = pd.DataFrame.from_dict(
        {'id': test_df['id'],
         'site_eui': grid_search.best_estimator_.predict(test_df)})

    # Check mean value of predictions
    print(f"Mean actuals: {round(train_df.site_eui.mean(), 2)}")
    print(f"Root mean squared actuals: {round(np.sqrt((train_df.site_eui ** 2).mean()), 2)}")
    print(f"Mean prediction: {round(predictions_df.site_eui.mean(), 2)}")
    print(f"Root mean squared prediction: {round(np.sqrt((predictions_df.site_eui ** 2).mean()), 2)}")
    print(f"Mean test: {72}")
    print(f"Root mean squared test: {94.6}")

    # Save local file
    save_local_submission(predictions_df, filename=out_file)
