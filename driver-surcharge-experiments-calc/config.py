import os
import numpy as np
import pandas as pd
import logging
import os
from dotenv import load_dotenv

CONFIDENCE_LEVEL = 0.99
BOOT_ITERATIONS = 1000
SIMULATION_NUM = 1000
PERCENT_SAMPLE = 0.5
MAX_N_SAMPLE = 1000
BUCKET_N = 1000
MIN_BUCKET_N = 100
SPLIT_COL = 'split'
USER_ID_COL = 'driver_id'
CORRECTION_TYPE = "fdr_bh"

base_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path="{dir}/.env".format(dir=base_path))  # custom settings
load_dotenv(dotenv_path="{dir}/.env.dist".format(dir=base_path))  # defaults


mysql_segmentation_conn = {
    'host': os.getenv("MYSQL_SEGMENTATION_HOST"),
    'port': int(os.getenv("MYSQL_SEGMENTATION_PORT")),
    'user': os.getenv("MYSQL_SEGMENTATION_USER"),
    'password': os.getenv("MYSQL_SEGMENTATION_PASSWORD"),
    'database': os.getenv("MYSQL_SEGMENTATION_DATABASE"),
}

mysql_db23_conn = {
    'host': os.getenv("MYSQL_DB23_HOST"),
    'port': int(os.getenv("MYSQL_DB23_PORT")),
    'user': os.getenv("MYSQL_DB23_USER"),
    'password': os.getenv("MYSQL_DB23_PASSWORD"),
    'database': os.getenv("MYSQL_DB23_DATABASE"),
}

mysql_prod_conn = {
    'host': os.getenv("MYSQL_PROD_HOST"),
    'port': int(os.getenv("MYSQL_PROD_PORT")),
    'user': os.getenv("MYSQL_PROD_USER"),
    'password': os.getenv("MYSQL_PROD_PASSWORD"),
    'database': os.getenv("MYSQL_PROD_DATABASE"),
}

_log_filename = os.getenv('LOG_FILENAME')
if len(_log_filename) > 0:
    if _log_filename[0] != '/' and _log_filename[0] != '~':
        _log_filename = base_path + '/' + _log_filename
    os.makedirs(os.path.dirname(_log_filename), exist_ok=True)
logging.basicConfig(
    filename=_log_filename,
    level=logging.getLevelName(os.getenv("LOG_LEVEL")),
)
version = os.getenv("SOURCE_CODE_VERSION")


SECONDARY_METRICS = {
    'drivers': ['driver_id', pd.Series.nunique],
    'trips': ['orders_count', np.sum],
    'gmv': ['driver_bill', np.sum],
    'payments': ['payment_id', pd.Series.nunique],
    'costs': ['fee', np.sum]
}


PRIMARY_METRICS = {
    "tpd": {
        "to_compress": True,
        "filter_to": None,
        "options": {
            "x": ['orders_count', np.sum],
            "y": ['driver_id', pd.Series.nunique],
            "n": ['driver_id', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": True
    },
    "average_bill": {
        "to_compress": True,
        "filter_to": None,
        "options": {
            "x": ['driver_bill', np.sum],
            "y": ['orders_count', np.sum],
            "n": ['driver_id', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": True
    },
    "cpt": {
        "to_compress": True,
        "filter_to": None,
        "options": {
            "x": ['fee', np.sum],
            "y": ['orders_count', np.sum],
            "n": ['driver_id', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": True
    }
}