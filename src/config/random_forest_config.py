from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression

random_forest_config = {

    # Data
    "target": "site_eui",
    "categorical_features": ['Year_Factor', 'State_Factor', 'building_class', 'facility_type',
                             'missing_energy_star_rating', 'missing_year_built', 'missing_direction_max_wind_speed',
                             # 'missing_direction_peak_wind_speed', 'missing_days_with_fog',
                             ],
    "numeric_features": ['floor_area',
                         'year_built',
                         'ELEVATION',
                         'energy_star_rating',
                         # 'Year_Mean',
                         ],
    "weather_features": ['january_min_temp',  # 'january_avg_temp', 'january_max_temp',
                         # 'february_min_temp', 'february_avg_temp', 'february_max_temp',
                         # 'march_min_temp', 'march_avg_temp', 'march_max_temp', 'april_min_temp',
                         # 'april_avg_temp', 'april_max_temp', 'may_min_temp', 'may_avg_temp',
                         # 'may_max_temp', 'june_min_temp', 'june_avg_temp', 'june_max_temp',
                         # 'july_min_temp', 'july_avg_temp',
                         'july_max_temp',  # 'august_min_temp',
                         # 'august_avg_temp', 'august_max_temp', 'september_min_temp',
                         # 'september_avg_temp', 'september_max_temp', 'october_min_temp',
                         # 'october_avg_temp', 'october_max_temp', 'november_min_temp',
                         # 'november_avg_temp', 'november_max_temp', 'december_min_temp',
                         # 'december_avg_temp', 'december_max_temp', 'cooling_degree_days',
                         # 'heating_degree_days', 'precipitation_inches', 'snowfall_inches',
                         # 'snowdepth_inches', 'days_below_30F', 'days_below_20F',
                         # 'days_below_10F', 'days_below_0F', 'days_above_80F', 'days_above_90F',
                         # 'days_above_100F', 'days_above_110F', 'direction_max_wind_speed',
                         # 'direction_peak_wind_speed', 'max_wind_speed', 'days_with_fog',
                         ],

    # Sklearn Pipeline
    "categorical_transformer": Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encode', OneHotEncoder(handle_unknown="ignore"))]),
    "numeric_transformer": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median')),
    ]),
    "weather_transformer": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler()),
        #    ("pca", PCA()),
    ]),
    # "feature_selector": SelectKBest(mutual_info_regression()),
    "estimator": RandomForestRegressor(),

    # Grid Search Cross-Validation
    "scoring": "neg_root_mean_squared_error",
    "param_grid": {'regressor__max_features': [None],
                   'regressor__max_depth': [5],
                   'regressor__min_samples_leaf': [3],
                   'regressor__criterion': ['squared_error'],
                   # "preprocessor__wea__pca__n_components": [10],
                   # "feature_selector__k": [5, 10, 20],
                   }
}
