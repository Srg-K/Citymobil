import datetime
import logging
import sys
import unittest

import numpy
import pandas

from citymobil_python_mysql_wrapper import MysqlWrapper


class TestMysqlWrapper(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logger = logging.getLogger()

        self._mysql = MysqlWrapper(
            logger,
            "local_mysql",
            # "localhost",
            3306,
            # 3316,
            "user",
            "password",
            "db"
        )

        self._table = "test_table"
        sql = """create table IF NOT EXISTS test_table (
                    `id` int(11) NOT NULL AUTO_INCREMENT,
                    `email` varchar(255) COLLATE utf8_bin NOT NULL,
                    `password` varchar(255) COLLATE utf8_bin NOT NULL,
                    `time` DATETIME NULL,
                    `number` int NULL,
                    PRIMARY KEY (`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
                AUTO_INCREMENT=1 ;"""
        self._mysql.execute(sql)

    def tearDown(self) -> None:
        sql = """drop table if exists test_table"""
        self._mysql.execute(sql)

    def test_insert_data_frame_1(self) -> None:
        # arrange
        data_frame = pandas.DataFrame({
            "email": ["a@a.a"],
            "password": ["a_password"],
        })
        expected = 1

        # action
        actual = self._mysql.insert_data_frame(self._table, data_frame)

        # assert
        self.assertEqual(expected, actual)

    def test_insert_rows_1(self) -> None:
        # arrange
        expected = 2

        # action
        actual = self._mysql.insert_rows(
            self._table,
            [
                ["a@a.a", "a_password"],
                ["b@b.b", "b_password"],
            ],
            ["email", "password"]
        )

        # assert
        self.assertEqual(expected, actual)

    def test_insert_rows_2_raise_on_empty_fields(self) -> None:
        # arrange
        # action
        # assert
        self.assertRaises(Exception,
                          self._mysql.insert_rows,
                          self._table,
                          [
                              ["a@a.a", "a_password"],
                              ["b@b.b", "b_password"],
                          ],
                          [],  # raise
                          )

    def test_insert_rows_3_raise_on_empty_rows(self) -> None:
        # arrange
        # action
        # assert
        self.assertRaises(Exception,
                          self._mysql.insert_rows,
                          self._table,
                          [
                              # raises
                          ],
                          ["email"],
                          )

    def test_insert_rows_4_commit_every_row(self) -> None:
        # arrange
        expected = 2

        # action
        actual = self._mysql.insert_rows(
            self._table,
            [
                ["a@a.a", "a_password"],
                ["b@b.b", "b_password"],
            ],
            ["email", "password"],
            commit_every=1,
        )

        # assert
        self.assertEqual(expected, actual)

    def test_insert_rows_5_serialize_datetime(self) -> None:
        # arrange
        expected = 2
        current_time = datetime.datetime.now().replace(microsecond=0)
        expected_timestamp = pandas.Timestamp(current_time)

        # action
        actual = self._mysql.insert_rows(
            self._table,
            [
                ["a@a.a", "a_password", current_time],
                ["b@b.b", "b_password", current_time],
            ],
            ["email", "password", "time"],
        )
        selected_data_frame = self._mysql.fetch("select * from {}".format(self._table))

        # assert
        self.assertEqual(expected, actual)
        self.assertEqual(expected_timestamp, selected_data_frame["time"][0])

    def test_insert_rows_6_serialize_np_nan(self) -> None:
        # arrange
        expected = 2

        # action
        actual = self._mysql.insert_rows(
            self._table,
            [
                ["a@a.a", "a_password", numpy.nan],
                ["b@b.b", "b_password", numpy.nan],
            ],
            ["email", "password", "time"],
        )

        # assert
        self.assertEqual(expected, actual)

    def test_fetch_1(self) -> None:
        # arrange
        expected = pandas.DataFrame({
            "id": [1],
            "email": ["a@a.a"],
            "password": ["a_password"],
            "time": [None],
            "number": [None],
        })
        self._mysql.insert_data_frame(self._table, expected)

        # action
        actual = self._mysql.fetch("select * from {}".format(self._table))
        print("expected:")
        print(expected)
        print("actual:")
        print(actual)

        # assert
        self.assertTrue(expected.equals(actual))
