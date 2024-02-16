import pandas as pd
import numpy as np
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

clickhouse_conn = {
    'url': os.getenv("CH_SEGMENTATION_URL"),
    'user': os.getenv("CH_SEGMENTATION_USER"),
    'password': os.getenv("CH_SEGMENTATION_PASSWORD")
}

exasol_conn = {
    'host': os.getenv("EXASOL_HOST"),
    'user': os.getenv("EXASOL_USER"),
    'password': os.getenv("EXASOL_PASSWORD"),
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

SPLIT_COL = 'split'
CONFIDENCE_LEVEL = 0.95
CORRECTION_TYPE = "fdr_bh"
BOOT_ITERATIONS = 1000
SIMULATION_NUM = 1
PERCENT_SAMPLE = 0.5
MAX_N_SAMPLE = 1000
BUCKET_N = 1000
MIN_BUCKET_N = 500
USER_ID_COL = "id_driver"
BOOT_IT = 500
BUFFER_AREA = 1000
POWER_LEVEL = 0.8


SECONDARY_METRICS = {
    'drivers': ['id_driver', pd.Series.nunique, ['retro == "tw"']],
    'drivers_is_geo': ['id_driver', pd.Series.nunique, ['is_geominimal_trip == True', 'is_ar == True', 'retro == "tw"']],
    # 'drivers_is_cnbl': ['id_driver', pd.Series.nunique, ['is_driver_in_cnbl_poly == True', 'is_ar == True', 'retro == "tw"']],
    # 'suggests_is_cnbl': ['fss_id', pd.Series.nunique, ['is_driver_in_cnbl_poly == True', 'retro == "tw"']],
    # 'orders_is_cnbl': ['order_id', pd.Series.nunique, ['is_driver_in_cnbl_poly == True', 'retro == "tw"']],
    # 'trips_is_cnbl': ['order_id', pd.Series.nunique, ['driver_picked_up_order == True', 'is_ar == True', 'is_driver_in_cnbl_poly == True', 'is_cp == True', 'retro == "tw"']],
    'orders': ['order_id', pd.Series.nunique, ['retro == "tw"']],
    'accepted_orders': ['order_id', pd.Series.nunique, ['is_ar == True', 'retro == "tw"']],
    'trips': ['order_id', pd.Series.nunique, ['is_cp == True', 'driver_picked_up_order == True', 'is_ar == True', 'retro == "tw"']],
    'trips_geominimal': ['order_id', pd.Series.nunique, ['is_geominimal_trip == True', 'driver_picked_up_order == True', 'is_ar == True', 'is_cp == True', 'retro == "tw"']],
    'suggests': ['fss_id', pd.Series.nunique, ['retro == "tw"']],
    'cost': ['damage', np.sum, ["driver_picked_up_order == True", "is_ar == True", 'retro == "tw"', 'is_cp == True', 'rn == 1']],
    'sh': ['supply_sec', lambda x: np.sum(x) / 3600, ['driver_picked_up_order == True', "is_ar == True", "is_cp == True", 'retro == "tw"', 'rn == 1']]
}

PRIMARY_METRICS = {
    "o2r": {
        "to_compress": True,
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['order_id', pd.Series.nunique, ['driver_picked_up_order == True', 'is_cp == True', 'is_ar == True']],
            "y": ['order_id', pd.Series.nunique],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    },
    "gtpd": {
        "to_compress": True,
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['order_id', pd.Series.nunique, ['is_geominimal_trip == True', 'driver_picked_up_order == True', 'is_ar == True', 'is_cp == True']],
            "y": ['id_driver', pd.Series.nunique],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    },
    "gtpgd": {
        "to_compress": True,
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['order_id', pd.Series.nunique, ['is_geominimal_trip == True', 'driver_picked_up_order == True', 'is_ar == True', 'is_cp == True']],
            "y": ['id_driver', pd.Series.nunique, ['is_geominimal_trip == True', 'driver_picked_up_order == True', 'is_ar == True']],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    },
    "rr": {
        "to_compress": True, 
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['id_driver', pd.Series.nunique, ['is_geominimal_trip == True', 'is_ar == True']],
            "y": ['id_driver', pd.Series.nunique],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    },
    "ar": {
        "to_compress": True, 
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['order_id', pd.Series.nunique, ['is_ar == True']],
            "y": ['order_id', pd.Series.nunique],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    },
    "tpgd": {
        "to_compress": True, 
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['order_id', pd.Series.nunique, ['is_geominimal_driver == True','driver_picked_up_order == True', 'is_cp == True', 'is_ar == True']],
            "y": ['id_driver', pd.Series.nunique, ['is_geominimal_trip == True', 'is_ar == True']],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    },
    "sh_per_driver": {
        "to_compress": True, 
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['supply_sec', lambda x: np.sum(x) / 3600, ['driver_picked_up_order == True', 'is_cp == True', 'is_ar == True', 'rn == 1']],
            "y": ['id_driver', pd.Series.nunique],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    },
    "cpt": {
        "to_compress": True, 
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['damage', np.sum, ['driver_picked_up_order == True', 'is_cp == True', 'is_ar == True', 'rn == 1']],
            "y": ['order_id', pd.Series.nunique, ['driver_picked_up_order == True', 'is_cp == True', 'is_ar == True']],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    },
    "o2r_wow": {
        "to_compress": True,
        "filter_to": None,
        "options": {
            "x": ['order_id', pd.Series.nunique, ['driver_picked_up_order == True', 'is_cp == True', 'is_ar == True']],
            "y": ['order_id', pd.Series.nunique],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": 'retro',
        "post_power": False
    },
    # "o2r_cnbl": {
    #     "to_compress": True,
    #     "filter_to": ['retro == "tw"'],
    #     "options": {
    #         "x": ['order_id', pd.Series.nunique, ['driver_picked_up_order == True', 'is_cp == True', 'is_ar == True', 'is_driver_in_cnbl_poly == True']],
    #         "y": ['order_id', pd.Series.nunique, ['is_driver_in_cnbl_poly == True']],
    #         "n": ['id_driver', pd.Series.nunique]
    #     },
    #     "estimator": "t_test_ratio",
    #     "metric_col": None,
    #     "split_col": SPLIT_COL,
    #     "post_power": False
    # },
    "of2r": {
        "to_compress": True, 
        "filter_to": ['retro == "tw"'],
        "options": {
            "x": ['order_id', pd.Series.nunique, ['driver_picked_up_order == True', 'is_cp == True', 'is_ar == True']],
            "y": ['fss_id', pd.Series.nunique],
            "n": ['id_driver', pd.Series.nunique]
        },
        "estimator": "t_test_ratio",
        "metric_col": None,
        "split_col": SPLIT_COL,
        "post_power": False
    }
    # ,
    # "of2r_cnbl": {
    #     "to_compress": True,
    #     "filter_to": ['retro == "tw"'],
    #     "options": {
    #         "x": ['order_id', pd.Series.nunique, ['driver_picked_up_order == True', 'is_cp == True', 'is_ar == True', 'is_driver_in_cnbl_poly == True']],
    #         "y": ['fss_id', pd.Series.nunique, ['is_driver_in_cnbl_poly == True']],
    #         "n": ['id_driver', pd.Series.nunique]
    #     },
    #     "estimator": "t_test_ratio",
    #     "metric_col": None,
    #     "split_col": SPLIT_COL,
    #     "post_power": False
    # }
}
