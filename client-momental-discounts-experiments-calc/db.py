import logging
import functools
import pyexasol
import citymobil_python_mysql_wrapper
import citymobil_python_clickhouse_wrapper
import config


@functools.lru_cache(None)
def get_mysql_conn_segm() -> citymobil_python_mysql_wrapper.MysqlWrapperInterface:
    return citymobil_python_mysql_wrapper.MysqlWrapper(
        logging.getLogger(),
        host=config.mysql_segmentation_conn['host'],
        port=config.mysql_segmentation_conn['port'],
        user=config.mysql_segmentation_conn['user'],
        password=config.mysql_segmentation_conn['password'],
        db=config.mysql_segmentation_conn['database'],
        charset='utf8mb4'
    )

@functools.lru_cache(None)
def get_mysql_conn_prod() -> citymobil_python_mysql_wrapper.MysqlWrapperInterface:
    return citymobil_python_mysql_wrapper.MysqlWrapper(
        logger=logging.getLogger(),
        host=config.mysql_prod_conn['host'],
        port=config.mysql_prod_conn['port'],
        user=config.mysql_prod_conn['user'],
        password=config.mysql_prod_conn['password'],
        db=config.mysql_prod_conn['database'],
        charset='utf8mb4'
    )

@functools.lru_cache(None)
def get_mysql_conn_db23() -> citymobil_python_mysql_wrapper.MysqlWrapperInterface:
    return citymobil_python_mysql_wrapper.MysqlWrapper(
        logger=logging.getLogger(),
        host=config.mysql_db23_conn['host'],
        port=config.mysql_db23_conn['port'],
        user=config.mysql_db23_conn['user'],
        password=config.mysql_db23_conn['password'],
        db=config.mysql_db23_conn['database'],
        charset='utf8mb4'
    )

@functools.lru_cache(None)
def get_ch_bi_conn() -> citymobil_python_clickhouse_wrapper.ClickHouseWrapperInterface:
    return citymobil_python_clickhouse_wrapper.ClickHouseWrapper(
        logging.getLogger(),
        url=config.ch_bi_conn['url'],
        user=config.ch_bi_conn['user'],
        password=config.ch_bi_conn['password'],
        create_and_assign_event_loop=False,  # for django threads with no event loop
        allow_nested_event_loops=True  # for jupyter threads with existed event loop
    )

@functools.lru_cache(None)
def get_ch_main_conn() -> citymobil_python_clickhouse_wrapper.ClickHouseWrapperInterface:
    return citymobil_python_clickhouse_wrapper.ClickHouseWrapper(
        logging.getLogger(),
        url=config.ch_main_conn['url'],
        user=config.ch_main_conn['user'],
        password=config.ch_main_conn['password'],
        create_and_assign_event_loop=False,  # for django threads with no event loop
        allow_nested_event_loops=True  # for jupyter threads with existed event loop
    )
