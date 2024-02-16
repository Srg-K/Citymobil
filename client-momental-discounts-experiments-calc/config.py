import pandas as pd
import logging
import os
from dotenv import load_dotenv


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

mysql_prod_conn = {
    'host': os.getenv("MYSQL_PROD_HOST"),
    'port': int(os.getenv("MYSQL_PROD_PORT")),
    'user': os.getenv("MYSQL_PROD_USER"),
    'password': os.getenv("MYSQL_PROD_PASSWORD"),
    'database': os.getenv("MYSQL_PROD_DATABASE"),
}

mysql_db23_conn = {
    'host': os.getenv("MYSQL_DB23_HOST"),
    'port': int(os.getenv("MYSQL_DB23_PORT")),
    'user': os.getenv("MYSQL_DB23_USER"),
    'password': os.getenv("MYSQL_DB23_PASSWORD"),
    'database': os.getenv("MYSQL_DB23_DATABASE"),
}

ch_bi_conn = {
    'url': os.getenv("CH_BI_URL"),
    'user': os.getenv("CH_BI_USER"),
    'password': os.getenv("CH_BI_PASSWORD"),
}

ch_main_conn = {
    'url': os.getenv("CH_MAIN_URL"),
    'user': os.getenv("CH_MAIN_USER"),
    'password': os.getenv("CH_MAIN_PASSWORD"),
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

SPLIT_COL = 'momental_group'
CONFIDENCE_LEVEL = 0.95
BOOT_ITERATIONS = 1000
SIMULATION_NUM = 500
PERCENT_SAMPLE = 0.5
MAX_N_SAMPLE = 1000
BUCKET_N = 1000
MIN_BUCKET_N = 500
USER_ID_COL = "id_client"
SPLIT_COL = "momental_group"
DT_START_TRESHOLD = '2021-07-01'

PARAMS = {
    'metrics': {
        "o2r": {
            "to_compress": True,
            "filter_to": None,
            "group_cols": ['momental_group', 'id_client'],
            "agg": lambda df: pd.Series({
                'x': df['rides'].sum(),
                'y': df['orders'].sum(),
                'n': df[USER_ID_COL].nunique()
            }),
            "estimator": "ttest_ratio",
            "metric_col": "metric",
            "split_col": SPLIT_COL,
            "user_id_col": USER_ID_COL,
            "post_power": True
        },
        "v2r": {
            "to_compress": True,
            "filter_to": None,
            "group_cols": ['momental_group', 'id_client'],
            "agg": lambda df: pd.Series({
                'x': df['rides'].sum(),
                'y': df['views'].sum(),
                'n': df[USER_ID_COL].nunique()
            }),
            "estimator": "ttest_ratio",
            "metric_col": "metric",
            "split_col": SPLIT_COL,
            "user_id_col": USER_ID_COL,
            "post_power": True
        },
        "v2o": {
            "to_compress": True,
            "filter_to": None,
            "group_cols": ['momental_group', 'id_client'],
            "agg": lambda df: pd.Series({
                'x': df['orders'].sum(),
                'y': df['views'].sum(),
                'n': df[USER_ID_COL].nunique()
            }),
            "estimator": "ttest_ratio",
            "metric_col": "metric",
            "split_col": SPLIT_COL,
            "user_id_col": USER_ID_COL,
            "post_power": True
        },
        "cpt": {
            "to_compress": True,
            "filter_to": None,
            "group_cols": ['momental_group', 'id_client'],
            "agg": lambda df: pd.Series({
                'x': df['sale_part_ride'].sum(),
                'y': df['rides'].sum(),
                'n': df[USER_ID_COL].nunique()
            }),
            "estimator": "ttest_ratio",
            "metric_col": "metric",
            "split_col": SPLIT_COL,
            "user_id_col": USER_ID_COL,
            "post_power": True
        },
        "tpс": {
            "to_compress": True,
            "filter_to": None,
            "group_cols": ['momental_group', 'id_client'],
            "agg": lambda df: pd.Series({
                'x': df['rides'].sum(),
                'y': df[USER_ID_COL].nunique(),
                'n': df[USER_ID_COL].nunique()
            }),
            "estimator": "ttest_ratio",
            "metric_col": "metric",
            "split_col": SPLIT_COL,
            "user_id_col": USER_ID_COL,
            "post_power": True
        }
    }
}
