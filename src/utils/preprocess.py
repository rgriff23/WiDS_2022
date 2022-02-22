import sys
import numpy as np
import pandas as pd

import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

from src.utils.io import save_local_data

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

FACILITY_MAP = {
    "Warehouse_Distribution_or_Shipping_center": "Warehouse_Distribution_or_Shipping_center",
    "Warehouse_Nonrefrigerated": "Warehouse_Nonrefrigerated",
    "Warehouse_Selfstorage": "Warehouse_Selfstorage",
    "Warehouse_Uncategorized": "Warehouse_Uncategorized",
    "Warehouse_Refrigerated": "Warehouse_Refrigerated",
    "Religious_worship": "Religious_worship",
    "Public_Assembly_Drama_theater": "Public_Assembly_Social_meeting",  # changed
    "Public_Assembly_Social_meeting": "Public_Assembly_Social_meeting",
    "Public_Assembly_Recreation": "Public_Assembly_Recreation",  # < 0.1%; added others to this group
    "Public_Assembly_Library": "Public_Assembly_Library",
    "Public_Assembly_Other": "Public_Assembly_Other",
    "Public_Assembly_Stadium": "Public_Assembly_Entertainment_culture",  # changed
    "Public_Assembly_Movie_Theater": "Public_Assembly_Entertainment_culture",  # changed
    "Public_Assembly_Entertainment_culture": "Public_Assembly_Entertainment_culture",
    "Public_Assembly_Uncategorized": "Public_Assembly_Social_meeting",  # changed
    "Public_Safety_Penitentiary": "Public_Assembly_Library",  # changed
    "Public_Safety_Courthouse": "Retail_Strip_shopping_mall",  # changed
    "Public_Safety_Uncategorized": "Retail_Strip_shopping_mall",  # changed
    "Public_Safety_Fire_or_police_station": "Public_Safety_Fire_or_police_station",
    "Grocery_store_or_food_market": "Grocery_store_or_food_market",
    "Retail_Enclosed_mall": "Retail_Enclosed_mall",
    "Retail_Strip_shopping_mall": "Retail_Strip_shopping_mall",
    "Retail_Vehicle_dealership_showroom": "Public_Assembly_Recreation",  # changed
    "Retail_Uncategorized": "Retail_Uncategorized",
    "Education_Other_classroom": "Education_Other_classroom",
    "Education_Uncategorized": "Education_Uncategorized",
    "Education_Preschool_or_daycare": "Education_Preschool_or_daycare",
    "Education_College_or_university": "Education_College_or_university",
    "Office_Uncategorized": "Office_Uncategorized",
    "Office_Medical_non_diagnostic": "Office_Medical_non_diagnostic",
    "Office_Bank_or_other_financial": "Office_Bank_or_other_financial",
    "Office_Mixed_use": "Office_Uncategorized",  # changed
    "Mixed_Use_Predominantly_Commercial": "Mixed_Use_Predominantly_Commercial",
    "Mixed_Use_Commercial_and_Residential": "Mixed_Use_Commercial_and_Residential",
    "Mixed_Use_Predominantly_Residential": "Multifamily_Uncategorized",  # changed
    "Commercial_Other": "Commercial_Other",
    "Commercial_Unknown": "Commercial_Unknown",
    "Service_Vehicle_service_repair_shop": "Service_Vehicle_service_repair_shop",
    "Service_Drycleaning_or_Laundry": "Public_Assembly_Recreation",  # changed
    "Service_Uncategorized": "Public_Assembly_Recreation",  # changed
    "Food_Service_Uncategorized": "Food_Sales",  # changed
    "Food_Service_Other": "Food_Sales",  # changed
    "Food_Service_Restaurant_or_cafeteria": "Food_Sales",  # changed
    "Food_Sales": "Food_Sales",  # < 0.1%; added others to this group
    "Nursing_Home": "Nursing_Home",
    "Health_Care_Uncategorized": "Lodging_Hotel",  # changed
    "Health_Care_Inpatient": "Health_Care_Inpatient",
    "Health_Care_Outpatient_Clinic": "Lodging_Hotel",  # changed
    "Health_Care_Outpatient_Uncategorized": "Lodging_Hotel",  # changed
    "Lodging_Other": "Multifamily_Uncategorized",  # changed
    "Lodging_Uncategorized": "Multifamily_Uncategorized",  # changed
    "Lodging_Dormitory_or_fraternity_sorority": "Lodging_Dormitory_or_fraternity_sorority",
    "Lodging_Hotel": "Lodging_Hotel",
    "Multifamily_Uncategorized": "Multifamily_Uncategorized",
    "2to4_Unit_Building": "2to4_Unit_Building",
    "5plus_Unit_Building": "5plus_Unit_Building",
    "Industrial": "Industrial",
    "Laboratory": "Laboratory",
    "Data_Center": "Laboratory",  # changed
    "Parking_Garage": "Parking_Garage"
}


def _missforest_imputation(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Missing values are filled for continuous features only.

    :param df:
    :param config: The column transformer used in the model.
    :return: Dataframe with the same shape as the input data
    """
    logging.info(f"Imputing missing values in numeric features with MissForest.")
    df_out = df.copy()

    # Convert categoricals to dummies
    numeric_features = config['numeric_features'] + config['weather_features']
    df_transformed = pd.concat([df[config['numeric_features']],
                                pd.get_dummies(df[config['categorical_features']])], axis=1)

    # Miss Forest imputation
    miss_forest = MissForest(criterion='squared_error', max_iter=10, n_jobs=-1)
    df_imputed = pd.DataFrame(miss_forest.fit_transform(df_transformed))
    df_imputed.columns = df_transformed.columns

    # Update the numeric columns only and return dataframe of same shape as the input
    df_out.loc[df_out.index, config['numeric_features']] = df_imputed[config['numeric_features']]
    return df_out


def preprocess_train_test(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Function to apply preprocessing steps to the train and test set.

    1. Override bad values with NaN
    2. Log transform skewed numerical features
    3. Add categorical features for missing numerical features
    4. Impute missing numerical features

    Returns preprocessed train and test sets.

    Also saves the intermediate files in the data directory so this step can
    be skipped to save time when iterating on downstream steps.
    """
    logging.info("************************")
    logging.info("*** PREPROCESS DATA ***")
    logging.info("************************\n")

    # Save target and original training data index
    target = train['site_eui']
    train_ix = train.index

    # Concatenate train (minus target) and test for combined preparation steps
    combo = pd.concat([train.drop('site_eui', axis=1), test]).reset_index(drop=True)
    logging.info(f"Combined train/test (minus target columns) into one dataframe: {combo.shape}.")

    # Override unreasonably old dates with nan
    logging.info(f"Replacing {sum(combo['year_built'] < 1500)} values where year_build < 1500 with NaN.")
    combo['year_built'] = np.where(combo['year_built'] < 1500, combo['year_built'], np.nan)

    # Log transform floor area
    logging.info(f"Log transforming floor_area.")
    combo['floor_area'] = combo['floor_area'].apply(np.log)

    # Modify "Year_Factor" to map to yearly means
    logging.info(f"Mapping Year_Factor to yearly means, including test set.")
    year_means = train.groupby(['Year_Factor']).mean()['site_eui']
    year_means = dict(zip(year_means.index, year_means.values))
    year_means[7] = 72
    combo['Year_Mean'] = combo['Year_Factor'].map(year_means)

    # Group facilities
    combo['facility_type'] = combo['facility_type'].map(FACILITY_MAP)

    # Add features for missing numericals
    logging.info(f"Adding categorical features for missingness in numerical columns: ")
    for col in train.isna().sum()[train.isna().sum() > 0].index.tolist():
        combo[f'missing_{col}'] = np.where(combo['energy_star_rating'].isna(), 'missing', 'present')
        logging.info(f"    - Added feature: {f'missing_{col}'}: {train[col].isna().sum()} missing values")

    # Missing value imputation
    #combo = _missforest_imputation(combo, config)

    # Split separated train+target and test dataframes
    train_out = pd.concat([combo.loc[train_ix], target], axis=1)
    test_out = combo[~combo.index.isin(train_ix)].reset_index(drop=True)

    save_local_data(train_out, 'preprocessed_train.csv')
    save_local_data(test_out, 'preprocessed_test.csv')
    logging.info(f"Finished preprocessing and saved preprocessed train and test data.\n")

    return train_out, test_out
