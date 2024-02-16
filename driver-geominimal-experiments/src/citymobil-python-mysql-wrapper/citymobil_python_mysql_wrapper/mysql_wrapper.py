from datetime import datetime
from abc import ABC
from contextlib import closing
from logging import Logger
from typing import Optional, List, Any

import numpy as np
import pandas
import pymysql
import newrelic.agent

from citymobil_python_mysql_wrapper.mysql_wrapper_interface import MysqlWrapperInterface


class MysqlWrapper(MysqlWrapperInterface, ABC):
    @newrelic.agent.background_task()
    def __init__(
            self,
            logger: Logger,
            host: str,
            port: int,
            user: str,
            password: str,
            db: Optional[str],
            charset: str = "utf8",
            use_unicode: bool = True
    ):
        self._logger = logger
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._db = db
        self._charset = charset
        self._use_unicode = use_unicode

        # test connection
        with closing(self._create_connection()):
            pass

    @newrelic.agent.background_task()
    def execute(self, query: str) -> int:
        self._logger.info("MysqlWrapper::execute(%s)", query)
        rows_affected = 0
        with closing(self._create_connection()) as mysql:
            with mysql.cursor() as cursor:
                # Create a new record
                rows_affected += cursor.execute(query)

            # connection is not autocommit by default. So you must commit to save
            # your changes.
            mysql.commit()
        return rows_affected

    @newrelic.agent.background_task()
    def fetch(self, query: str) -> pandas.DataFrame:
        self._logger.info("MysqlWrapper::fetch(%s)", query)
        result: List[Any] = []
        with closing(self._create_connection()) as mysql:
            with mysql.cursor() as cursor:
                # Read a single record
                cursor.execute(query)
                result = cursor.fetchall()
        return pandas.DataFrame(result)

    @newrelic.agent.background_task()
    def insert_data_frame(
            self,
            table: str,
            data_frame: pandas.DataFrame) -> int:
        self._logger.info("MysqlWrapper::insert_data_frame(%s)", table)
        rows = data_frame.values.tolist()
        fields = data_frame.columns.tolist()
        return int(self.insert_rows(table, rows, fields))

    @newrelic.agent.background_task()
    def insert_rows(
            self,
            table: str,
            rows: List[Any],
            fields: List[str],
            commit_every: int = 1000) -> int:
        self._logger.info("MysqlWrapper::insert_rows(%s)", table)
        rows_affected = 0
        if fields:
            target_fields = ", ".join(fields)
            target_fields = "({})".format(target_fields)
        else:
            raise Exception("Empty fields")

        if len(rows) == 0:
            raise Exception(f"No rows to insert in table {table}")

        with closing(self._create_connection()) as mysql:
            mysql.commit()

            with closing(mysql.cursor()) as cur:
                for i, row in enumerate(rows, 1):
                    cells: List[str] = []
                    for cell in row:
                        cells.append(self._serialize_cell(cell))
                    sql = "INSERT INTO {0} {1} VALUES ({2});".format(
                        table,
                        target_fields,
                        ",".join(cells))
                    rows_affected += cur.execute(sql)
                    if i % commit_every == 0:
                        mysql.commit()
                        self._logger.info(
                            f"Loaded {i} into {table} rows so far")

            mysql.commit()
        self._logger.info(f"Done loading. Loaded a total of {len(rows)} rows")
        return rows_affected

    @staticmethod
    def _serialize_cell(cell: Any) -> str:
        """
        serialize everything to string
        """
        if isinstance(cell, datetime):
            return "'{}'".format(cell.isoformat())
        # if isinstance(cell, geometry.polygon.Polygon):
        #     return "ST_GEOMFROMTEXT('{}')".format(cell)
        if isinstance(cell, str):
            return "'{}'".format(cell)
        if cell is None:
            return 'NULL'
        if np.isnan(cell):
            return 'NULL'
        return str(cell)

    @newrelic.agent.background_task()
    def _create_connection(self) -> pymysql.connections.Connection:
        return pymysql.connect(host=self._host,
                               port=self._port,
                               user=self._user,
                               password=self._password,
                               db=self._db,
                               charset=self._charset,
                               cursorclass=pymysql.cursors.DictCursor)
