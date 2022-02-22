import os
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(ROOT_DIR, "data")
OUT_DIR = Path(ROOT_DIR, "out")


def load_train_test() -> (pd.DataFrame, pd.DataFrame):
    """
    Load train and test data from local project data directory.
    :return:
    """
    logging.info("*****************")
    logging.info("*** LOAD DATA ***")
    logging.info("*****************\n")

    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    logging.info(f"Loaded training data: {train.shape}")

    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    logging.info(f"Loaded test data: {test.shape}\n")

    return train, test


def load_local_data(filename: str):
    return pd.read_csv(Path(DATA_DIR, filename))


def load_local_predictions(filename: str):
    return pd.read_csv(Path(OUT_DIR, filename))


def save_local_data(df: pd.DataFrame, filename: str):
    try:
        df.to_csv(Path(DATA_DIR, filename), index=False)
        logging.info(f"Saved data with shape {df.shape} to {DATA_DIR}/{filename}")
    except:
        logging.warning(f"Failed to write local file {filename} to {DATA_DIR}.")


def save_local_submission(df: pd.DataFrame, filename: str):
    # Ensure valid columns for competition
    assert df.columns.tolist() == ['id', 'site_eui']

    try:
        df.to_csv(Path(OUT_DIR, filename), index=False)
        logging.info(f"Saved data with shape {df.shape} to {OUT_DIR}/{filename}")
    except:
        logging.warning(f"Failed to write local file.")


def submit_to_competition(filename: str, competition: str, message: str = None):
    # logging.warning("Programmatic submission to Kaggle not yet implemented.")
    # Check previous submissions
    os.system(f"kaggle competitions submissions -c {competition}")
    try:
        file = Path(OUT_DIR, filename)
        print(file)
        if message is None:
            message = f"Submitted file '{filename}' to competition {competition}."
        cmd = f"kaggle competitions submit -c {competition} -f {file} -m {message}"
        os.system(cmd)
        logging.info(f"Submitted file {filename} to competition {competition}.")
    except:
        logging.warning(f"Failed to submit file {filename} to competition {competition}.")
