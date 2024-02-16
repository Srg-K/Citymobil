# pylint: skip-file
import datetime
import logging

import pandas

from citymobil_python_clickhouse_wrapper import ClickHouseWrapper

# initialize
logger = logging.getLogger()
clickhouse = ClickHouseWrapper(
    logger,
    url="http://local_clickhouse:8123",
    user="default",
    password="",
    create_and_assign_event_loop=False,  # for django threads with no event loop
    allow_nested_event_loops=False  # for jupyter threads with existed event loop
)

# select pandas data frame
df = clickhouse.fetch("select 1")
print("select 1=\n", df)

# create table
clickhouse.execute("drop database IF EXISTS dynamic_configs")
clickhouse.execute("create database IF NOT EXISTS dynamic_configs")
clickhouse.execute(
    """create table IF NOT EXISTS dynamic_configs.polygon_coefficient_model_assigner
(
name_polygon String,
polygon_coordinates String,
is_enabled UInt8,
id_coefficient_model UInt16,
date_time Nullable(DATETIME)
) ENGINE = MergeTree()
ORDER BY name_polygon""")

# insert rows
clickhouse.insert_rows(
    "dynamic_configs.polygon_coefficient_model_assigner",
    [
        [
            7032,
            1,
            'Ярославль',
            '[["57.561711213773","39.900267722473"],["57.574809221059","39.843319060669"],["57.582769675412","39.824032557491"],["57.590528940757","39.815934258294"],["57.596067025993","39.80939601873"],["57.598240463293","39.802256964346"],["57.614161005939","39.784417843584"],["57.621075532296","39.808385970459"],["57.644035956353","39.797286239585"],["57.669306442053","39.770534636841"],["57.68926109648","39.751051070557"],["57.695695729769","39.764268996582"],["57.704426611603","39.756715895997"],["57.729642395475","39.746544959412"],["57.725831304501","39.773452880249"],["57.706101130076","39.786546045488"],["57.711910130517","39.811140522984"],["57.724076386327","39.806942789742"],["57.731188531571","39.818194580426"],["57.734075922234","39.834596212419"],["57.719729670077","39.848361004549"],["57.705692615257","39.787056511438"],["57.687972891299","39.818964146013"],["57.669527971052","39.77137118152"],["57.643933405673","39.797721507416"],["57.651089347003","39.901254775391"],["57.66544297715","39.965112807617"],["57.653482014618","39.983308913574"],["57.6277064932","39.973352553711"],["57.620131616125","39.906249301048"],["57.570612839423","39.942539336547"],["57.551607759066","39.980219008789"],["57.546301165283","39.970691802368"],["57.542193790506","39.971206786499"],["57.54511034677","39.927187498253"],["57.561711213773","39.900267722473"]]',
            datetime.datetime.now(),
        ]
    ],
    [
        "id_coefficient_model",
        "is_enabled",
        "name_polygon",
        "polygon_coordinates",
        "date_time"
    ]
)

# insert pandas data frame
clickhouse.insert_data_frame(
    "dynamic_configs.polygon_coefficient_model_assigner",
    pandas.DataFrame({
        "date_time": [None, None],
        "id_coefficient_model": [7032, 7033],
        "is_enabled": [1, 0],
        "name_polygon": ['Ярославль', 'Москва'],
        "polygon_coordinates": ['с1', 'c2'],
    }))

# select all
all_df = clickhouse.fetch(
    "select * from dynamic_configs.polygon_coefficient_model_assigner")
print("select * from dynamic_configs.polygon_coefficient_model_assigner=\n", all_df)

# drop table
clickhouse.execute(
    "drop table IF EXISTS dynamic_configs.polygon_coefficient_model_assigner")
