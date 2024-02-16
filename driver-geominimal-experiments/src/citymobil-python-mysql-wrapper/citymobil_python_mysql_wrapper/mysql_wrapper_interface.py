import abc
from typing import List, Any

import pandas


class MysqlWrapperInterface(abc.ABC):
    @abc.abstractmethod
    def execute(self, query: str) -> int:
        pass

    @abc.abstractmethod
    def fetch(self, query: str) -> pandas.DataFrame:
        pass

    @abc.abstractmethod
    def insert_data_frame(
            self,
            table: str,
            data_frame: pandas.DataFrame) -> int:
        pass

    @abc.abstractmethod
    def insert_rows(
            self,
            table: str,
            rows: List[Any],
            fields: List[str],
            commit_every: int = 1000) -> int:
        pass
