import functools
import pymysql
import citymobil_python_mysql_wrapper
import logging

import config


@functools.lru_cache(None)
def get_mysql_conn_segm() -> citymobil_python_mysql_wrapper.MysqlWrapperInterface:
    return citymobil_python_mysql_wrapper.MysqlWrapper(
        logger=logging.getLogger(),
        host=config.mysql_segmentation_conn['host'],
        port=config.mysql_segmentation_conn['port'],
        user=config.mysql_segmentation_conn['user'],
        password=config.mysql_segmentation_conn['password'],
        db=config.mysql_segmentation_conn['database'],
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

