import logging
import functools
import citymobil_python_mysql_wrapper
import citymobil_python_clickhouse_wrapper
import config_connect

@functools.lru_cache(None)
def get_mysql_conn() -> citymobil_python_mysql_wrapper.MysqlWrapperInterface:
    return citymobil_python_mysql_wrapper.MysqlWrapper(
        logging.getLogger(),
        host = config_connect.mysql_segmentation_conn['host'],
        port = config_connect.mysql_segmentation_conn['port'],
        user = config_connect.mysql_segmentation_conn['user'],
        password = config_connect.mysql_segmentation_conn['password'],
        db = config_connect.mysql_segmentation_conn['database'],
    )
@functools.lru_cache(None)
def get_ch_main_conn() -> citymobil_python_clickhouse_wrapper.ClickHouseWrapperInterface:
    return citymobil_python_clickhouse_wrapper.ClickHouseWrapper(
        logging.getLogger(),
        url=config_connect.ch_main_conn['url'],
        user=config_connect.ch_main_conn['user'],
        password=config_connect.ch_main_conn['password'],
        create_and_assign_event_loop=False,  # for django threads with no event loop
        allow_nested_event_loops=True  # for jupyter threads with existed event loop
    )
