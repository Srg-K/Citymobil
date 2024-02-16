import logging
import functools
import pyexasol

import citymobil_python_mysql_wrapper

import config


@functools.lru_cache(None)
def get_mysql_conn_segm() -> citymobil_python_mysql_wrapper.MysqlWrapperInterface:
    return citymobil_python_mysql_wrapper.MysqlWrapper(
        logging.getLogger(),
        config.mysql_segmentation_conn['host'],
        config.mysql_segmentation_conn['port'],
        config.mysql_segmentation_conn['user'],
        config.mysql_segmentation_conn['password'],
        config.mysql_segmentation_conn['database'],
    )


@functools.lru_cache(None)
def get_mysql_conn_prod() -> citymobil_python_mysql_wrapper.MysqlWrapperInterface:
    return citymobil_python_mysql_wrapper.MysqlWrapper(
        logging.getLogger(),
        config.mysql_prod_conn['host'],
        config.mysql_prod_conn['port'],
        config.mysql_prod_conn['user'],
        config.mysql_prod_conn['password'],
        config.mysql_prod_conn['database'],
    )


def get_exasol_connection():
    return pyexasol.connect(
        dsn=config.exasol_conn['host'],
        user=config.exasol_conn['user'],
        password=config.exasol_conn['password'],
        fetch_dict=True,
    )
