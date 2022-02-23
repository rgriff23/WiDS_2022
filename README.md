# Introduction

Kaggle competition: https://www.kaggle.com/c/widsdatathon2022

The scripts in this repo produce CSV files with predictions which
can be uploaded to Kaggle.

### Project structure

- `data`: Input data and intermediate files for modeling
- `out`: Prediction files
- `src`: Source code
    - `utils`: Utility functions used by training scripts
    - `config`: Configurations used by training scripts
    - `run_pipeline.py`: Training script which takes data and a model config as input 
      and writes predictions to the `out` directory
    - `compute_final_predictions.py`: Imports one or more prediction dataframes, combines
      the outputs, performs final post-processing adjustments, and writes final predictions.
      
### Usage

Run the `run_pipeline` script to generate predictions using a model config.
Additional model configs can be created and imported into this script to run
different models and save the prediction files with different names in the `out` folder.

Run the `compute_final_predictions` script to perform post-processing such as
combining predictions from multiple models and adjusting predictions to have 
the same mean as the test set. This script outputs a "final" prediction dataframe.

### Notes on models

- Random Forest is more accurate with most weather variables excluded
    - PCA and feature selection routines were not helpful
    - Manually adjusting the mean prediction to match the test set was helpful
- XGBoost significantly outperforms Random Forest and has several additional advantages
    - Imputes missing values automatically 
    - Results are close to the average of the test set without need for manual adjustment
- UMAP dimension reduction on weather variables is helpful