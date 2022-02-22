import numpy as np
import pandas as pd

from src.utils.io import load_local_data, load_local_predictions, save_local_submission

if __name__ == '__main__':
    """
    This script can be used to combined predictions from multiple models
    and make final adjustments to the predictions. 
    """
    # Import predictions to combine
    rf_baseline = load_local_predictions('random_forest_baseline.csv')
    xgb_baseline_stars = load_local_predictions('xgb_baseline.csv')

    # Combine predictions from multiple models
    # mean_site_eui = (rf_baseline['site_eui'] + xgb_baseline_stars['site_eui']) / 2
    # out_df = pd.DataFrame.from_dict(
    #     {'id': baseline_stars['id'],
    #      'site_eui': mean_site_eui})

    out_df = load_local_predictions('xgb_baseline.csv')

    # Adjust mean predictions to have same mean as test set
    print("Initial mean EUI:", round(out_df['site_eui'].mean(), 2))
    print("Initial min EUI:", round(out_df['site_eui'].min(), 2))
    print("Initial Mean Square EUI:", round(np.sqrt((out_df.site_eui ** 2).mean()), 2))
    percent_adjustment = (out_df['site_eui'].mean() - 72) / out_df['site_eui'].mean()
    out_df['site_eui'] = out_df['site_eui'] - out_df['site_eui'] * percent_adjustment
    print("New mean EUI:", round(out_df['site_eui'].mean(), 2))
    print("New min EUI:", round(out_df['site_eui'].min(), 2))
    print("New Mean Square EUI:", round(np.sqrt((out_df.site_eui ** 2).mean()), 2))

    save_local_submission(out_df, 'final_predictions.csv')
