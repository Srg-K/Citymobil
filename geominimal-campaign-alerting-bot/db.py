import functools
import pyexasol
import config
import logging
import citymobil_python_clickhouse_wrapper

@functools.lru_cache(None)
def get_exasol_connection():
    return pyexasol.connect(
        dsn=config.exasol_conn['host'],
        user=config.exasol_conn['user'],
        password=config.exasol_conn['password'],
        fetch_dict=True,
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
