import abc
from typing import List, Any

import pandas as pd


class ClickHouseWrapperInterface(abc.ABC):
    @abc.abstractmethod
    def execute(self, query: str) -> None:
        pass

    @abc.abstractmethod
    def fetch(self, query: str) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_pandas_df(self, query: str) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def insert_data_frame(
            self,
            table: str,
            data_frame: pd.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def insert_df(
            self,
            table: str,
            data_frame: pd.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def insert_rows(
            self,
            table: str,
            rows: List[Any],
            fields: List[str]) -> None:
        pass
