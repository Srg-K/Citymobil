import logging
import functools

import citymobil_python_clickhouse_wrapper
import citymobil_python_mysql_wrapper

import config


@functools.lru_cache(None)
def get_mysql_conn() -> citymobil_python_mysql_wrapper.MysqlWrapperInterface:
    return citymobil_python_mysql_wrapper.MysqlWrapper(
        logging.getLogger(),
        config.mysql_segmentation_conn['host'],
        config.mysql_segmentation_conn['port'],
        config.mysql_segmentation_conn['user'],
        config.mysql_segmentation_conn['password'],
        config.mysql_segmentation_conn['database'],
    )


@functools.lru_cache(None)
def get_ch_seg_conn() -> citymobil_python_clickhouse_wrapper.ClickHouseWrapperInterface:
    return citymobil_python_clickhouse_wrapper.ClickHouseWrapper(
        logging.getLogger(),
        url=config.ch_segmentation_conn['url'],
        user=config.ch_segmentation_conn['user'],
        password=config.ch_segmentation_conn['password'],
        create_and_assign_event_loop=False,  # for django threads with no event loop
        allow_nested_event_loops=True  # for jupyter threads with existed event loop
    )

