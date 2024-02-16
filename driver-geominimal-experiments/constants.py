BOOT_IT = 500
BUFFER_AREA = 1000
CONFIDENCE_LEVEL = 0.05
POWER_LEVEL = 0.8

METRIC_DICT = {
    "rr": {
        "x": "id_driver",
        "y": "id_driver",
        "x_condition": ['is_geominimal == True', 'is_ar == True', 'retro == "tw"'],
        "y_condition": ['retro == "tw"'],
        "x_f": lambda x: x.nunique(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    },
    "ar": {
        "x": "order_id",
        "y": "order_id",
        "x_condition": ['is_ar == True', 'retro == "tw"'],
        "y_condition": ['retro == "tw"'],
        "x_f": lambda x: x.nunique(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    },
    "tpgd": {
        "x": "order_id",
        "y": "id_driver",
        "x_condition": ['is_cp == True', 'retro == "tw"'],
        "y_condition": ['is_geominimal == True', 'is_ar == True', 'retro == "tw"'],
        "x_f": lambda x: x.nunique(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    },
    "sh_per_driver": {
        "x": "supply_sec",
        "y": "id_driver",
        "x_condition": ['retro == "tw"'],
        "y_condition": ['retro == "tw"'],
        "x_f": lambda x: x.sum() / 3600 ,
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    },
    "cpt": {
        "x": "damage",
        "y": "order_id",
        "x_condition": ['is_cp == True','retro == "tw"'],
        "y_condition": ['is_cp == True','retro == "tw"'],
        "x_f": lambda x: x.sum(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    },
    "o2r_wow": {
        "x": "order_id",
        "y": "order_id",
        "x_condition": ['is_cp == True'],
        "y_condition": None,
        "x_f": lambda x: x.nunique(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "retro"
    },
    "o2r_cnbl": {
        "x": "order_id",
        "y": "order_id",
        "x_condition": ['is_cp == True', 'is_driver_in_cnbl_poly == True','retro == "tw"'],
        "y_condition": ['is_driver_in_cnbl_poly == True','retro == "tw"'],
        "x_f": lambda x: x.nunique(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    },
    "o2r": {
        "x": "order_id",
        "y": "order_id",
        "x_condition": ['is_cp == True','retro == "tw"'],
        "y_condition": ['retro == "tw"'],
        "x_f": lambda x: x.nunique(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    },
    "of2r": {
        "x": "order_id",
        "y": "fss_id",
        "x_condition": ['is_cp == True','retro == "tw"'],
        "y_condition": ['retro == "tw"'],
        "x_f": lambda x: x.nunique(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    },
    "of2r_cnbl": {
        "x": "order_id",
        "y": "fss_id",
        "x_condition": ['is_cp == True', 'is_driver_in_cnbl_poly == True','retro == "tw"'],
        "y_condition": ['is_driver_in_cnbl_poly == True','retro == "tw"'],
        "x_f": lambda x: x.nunique(),
        "y_f": lambda x: x.nunique(),
        "user_level_col": "id_driver",
        "split": "split"
    }
}