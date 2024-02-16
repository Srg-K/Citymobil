import datetime
import sys

import pandas

from citymobil_python_mysql_wrapper import MysqlWrapper
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger()
mysql = MysqlWrapper(
    logger,
    "local_mysql",
    # "localhost",
    3306,
    # 3316,
    "user",
    "password",
    "db"
)

# .execute
sql = """create table IF NOT EXISTS test_table (
                    `id` int(11) NOT NULL AUTO_INCREMENT,
                    `email` varchar(255) NOT NULL,
                    `password` varchar(255) NOT NULL,
                    `time` DATETIME NOT NULL,
                    PRIMARY KEY (`id`)
                ) ENGINE=InnoDB 
                AUTO_INCREMENT=1 ;"""
mysql.execute(sql)

# .insert_data_frame
data_frame = pandas.DataFrame({
    "email": ["a@a.a"],
    "password": ["a_password"],
    "time": datetime.datetime.now()
})
mysql.insert_data_frame("test_table", data_frame)

# .insert_rows
mysql.insert_rows(
    "test_table",
    [
        ["b@b.b", "b_password", datetime.datetime.now()],
    ],
    ["email", "password", "time"]
)

# .fetch
data_frame = mysql.fetch("select * from {}".format("test_table"))
print(data_frame)
