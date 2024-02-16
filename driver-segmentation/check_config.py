import logging

from citymobil_python_clickhouse_wrapper import ClickHouseWrapper
from citymobil_python_mysql_wrapper import MysqlWrapper

import config
import db

logger = logging.getLogger()

print('Checking MySQL segmentation connection and tables...')
tables = list(map(lambda row: row[0], db.get_mysql_conn().fetch("SHOW TABLES").values))
assert 'driver_segmentation_dates' in tables
assert 'driver_segmentation_rfm' in tables
print('OK')


print('Checking ClickHouse segmentation connection and tables...')
tables = list(map(
    lambda row: row[0],
    db.get_ch_seg_conn().fetch(
        "SELECT name FROM system.tables WHERE database='{db}'".format(db=config.ch_segmentation_conn['database'])
    ).values
))
assert 'drivers' in tables
assert 'orders_dist' in tables
assert 'driver_payments_dist' in tables
print('OK')


# print('Validating CSV...')
# @todo
print('OK')
