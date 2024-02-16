import asyncio
import logging
from datetime import datetime
from logging import Logger
from typing import List, Any, Optional

import nest_asyncio
import newrelic.agent
import numpy as np
import pandas
from aiochclient import ChClient
from aiohttp import ClientSession

from citymobil_python_clickhouse_wrapper.clickhouse_wrapper_interface import ClickHouseWrapperInterface


class ClickHouseWrapper(ClickHouseWrapperInterface):
    def __init__(
            self,
            logger: Optional[Logger],
            url: str,
            user: str,
            password: str,
            # for django threads with no event loop
            create_and_assign_event_loop: bool = False,
            # for jupyter threads with existed event loop
            allow_nested_event_loops: bool = False
    ) -> None:
        if logger is None:
            # init logger with null handler
            self._logger = logging.getLogger(__name__)
            self._logger.addHandler(logging.NullHandler())
        else:
            self._logger = logger

        if create_and_assign_event_loop:
            asyncio.set_event_loop(asyncio.new_event_loop())

        if allow_nested_event_loops:
            nest_asyncio.apply()

        self._url = url
        self._user = user
        self._password = password

    @newrelic.agent.background_task()
    def execute(self, query: str) -> None:
        self._logger.info("ClickHouseWrapper::execute(%s)", query)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._execute_query(query))

    @newrelic.agent.background_task()
    def fetch(self, query: str) -> pandas.DataFrame:
        self._logger.info("ClickHouseWrapper::fetch(%s)", query)
        loop = asyncio.get_event_loop()
        return pandas.DataFrame(
            loop.run_until_complete(
                self._fetch_query(query)))

    @newrelic.agent.background_task()
    def get_pandas_df(self, query: str) -> pandas.DataFrame:
        return self.fetch(query)

    @newrelic.agent.background_task()
    def insert_data_frame(
            self,
            table: str,
            data_frame: pandas.DataFrame) -> None:
        self._logger.info("ClickHouseWrapper::insert_data_frame(%s)", table)
        rows = data_frame.values.tolist()
        fields = data_frame.columns.tolist()
        self.insert_rows(table, rows, fields)

    @newrelic.agent.background_task()
    def insert_df(
            self,
            table: str,
            data_frame: pandas.DataFrame) -> None:
        return self.insert_data_frame(table, data_frame)

    @newrelic.agent.background_task()
    def insert_rows(
            self,
            table: str,
            rows: List[Any],
            fields: List[str]) -> None:
        self._logger.info("ClickHouseWrapper::insert_rows(%s)", table)

        if len(rows) == 0:
            raise Exception(f"No rows to insert in table {table}")
        if not fields:
            raise Exception("Empty fields")

        target_fields_str = ", ".join(fields)
        target_fields_str = "({})".format(target_fields_str)

        cells_strings: List[str] = []
        for row in rows:
            cur_row_str = "({0})".format(
                ",".join([self._serialize_cell_str(cell) for cell in row])
            )
            cells_strings.append(
                cur_row_str
            )

        sql = "INSERT INTO {0} {1} VALUES {2}".format(
            table,
            target_fields_str,
            ",".join(cells_strings))

        self.execute(sql)

    async def _execute_query(self, query: str) -> None:
        async with ClientSession() as session:
            client = ChClient(
                session,
                url=self._url,
                user=self._user,
                password=self._password)
            await client.execute(query)

    async def _fetch_query(self, query: str) -> List[Any]:
        async with ClientSession() as session:
            client = ChClient(
                session,
                url=self._url,
                user=self._user,
                password=self._password)
            result = await client.fetch(query)
            return list(result)

    @staticmethod
    def _serialize_cell_str(cell: Any) -> str:
        """
        serialize everything to string
        """
        if isinstance(cell, datetime):
            return "'{}'".format(cell.replace(microsecond=0).isoformat())
        # if isinstance(cell, geometry.polygon.Polygon):
        #     return "ST_GEOMFROMTEXT('{}')".format(cell)
        if isinstance(cell, str):
            return "'{}'".format(cell)
        if cell is None:
            return 'NULL'
        if np.isnan(cell):
            return 'NULL'
        return str(cell)
