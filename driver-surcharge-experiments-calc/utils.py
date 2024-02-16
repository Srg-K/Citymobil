import pandas as pd
import numpy as np
import config
import itertools
import math
import db
import query as q
import logging
from datetime import datetime


def eval_condition(input: [pd.DataFrame, pd.Series], condition: list, df: pd.DataFrame):
    """
    Фильтрация таблицы или колонки

    Parameters
    ----------
    input : [pd.DataFrame, pd.Series]
        Таблица или колонка. Что будет подано на вход – то и будет отфильтровано.
    condition : array_like or None
        Перечисление условий для фильтрации. При None фильтрация выключена
    df : pd.DataFrame
        Таблица с сырыми данными эксперимента для фильтрации

    Returns
    -------
    out : pd.Series or pd.DataFrame
        Отфильтрованная таблица или колонка.
        Если ``condition`` с ``None``, вернется то что подавалось на input
    """
    if condition is None:
        return input

    else:
        if isinstance(input, pd.DataFrame): instns = 'input'
        elif isinstance(input, pd.Series): instns = 'df'

        conditions = []
        for i in range(len(condition)):
            conditions.append(f'({instns}.' + condition[i] + ')')
        conditions = ' & '.join(conditions)
        return eval('input[(' + conditions + ')]')


def get_raw_data(
        sur_id: int,
        date_start,
        date_end
):
    """
    Получение сырых данных: оплаты по гд, сплиты водителей

    Parameters
    ----------
    sur_id
    date_start
    date_end

    Returns
    -------
    df
    """
    dfs = []
    dt = datetime.now().strftime("%Y-%m-%d")
    experiment_period = pd.date_range(date_start, min(date_end.strftime("%Y-%m-%d"), dt))
    payments = db.get_mysql_conn_prod().fetch(q.q_2(sur_id))
    for single_date in experiment_period:
        res = db.get_mysql_conn_db23().fetch(q.q_3(sur_id, single_date))
        dfs.append(res)
    try:
        drivers = pd.concat(dfs)
        df = payments.merge(drivers, on=['driver_id', 'date'], how='left')
        return df
    except Exception as e:
        return None



def as_buckets(df: pd.DataFrame, bucket_n: int):
    """
    Создание бакетов

    Присваивает пользователям уникальный id бакета.

    Parameters
    ----------
    df
    bucket_n : int
        Количество бакетов. Регулируется в config.py

    Returns
    -------
    out : list
        Оптимальное колиество бакетов для метрики
    """
    users_distinct = df[config.USER_ID_COL].unique()
    df_ = pd.DataFrame({
        config.USER_ID_COL: users_distinct,
        "bucket": [i % bucket_n for i in range(0, len(users_distinct))]
    })
    df = df.merge(df_, on=config.USER_ID_COL)
    return df


def parse_options(df, options):
    """
    Обертка для парсинга options из конфига метрик

    Parameters
    ----------
    df
    options

    Returns
    -------

    """

    elem = options
    if len(elem) < 3: elem.append(None)
    elem_col, elem_f, elem_condition = elem
    return elem_f(eval_condition(df[elem_col], elem_condition, df))


def compress_and_filter(
        df: pd.DataFrame,
        to_compress: int,
        estimator: str,
        options: dict = None,
        filter_to=None,
        split_col: str = config.SPLIT_COL):
    """
    Агрегация метрик по бакетам.

    По сути служит для трансформации метрик в ratio формат
    Помимо этого применяет все фильтры из конфига метрика

    Parameters
    ----------
    df
    split_col : str
        Колонка со сплитами
    estimator :
        Статистический оценщик
    to_compress : bool
        Флаг необходимости компрессии
    filter_to : list
        Список фильтров
    options : dict
        Предварительная агрегация перед присвоением бакета

    Returns
    -------
    out : pd.DataFrame
        колонки:
        x - числитель,
        y - знаменатель,
        n - кол-во наблюдений для сохранения веса ratio

    """
    if filter_to is not None:
        df = eval_condition(df, filter_to, df)

    bucket_n = min(len(df) / 2, config.BUCKET_N)
    if len(df) < 100:
        bucket_n = min(len(df) / 2, config.MIN_BUCKET_N)
    df = as_buckets(df, bucket_n)

    if to_compress == 1 or to_compress:
        grp_col = 'bucket'
    elif (to_compress == 0 or not to_compress) & (estimator == 't_test_ratio'):
        grp_col = options["n"][0]
    else:
        return df

    df = df.groupby([split_col, grp_col]).apply(
        lambda df: pd.Series({
            "x": parse_options(df, options["x"]),
            "y": parse_options(df, options["y"]),
            "n": parse_options(df, options["n"])
        })).reset_index()
    return df


def parse_secondary(df, key):
    """Парсинг конфига показателей из SECONDARY_METRICS"""
    options = config.SECONDARY_METRICS[key]

    if len(options) < 3: options.append(None)
    col, f, condition = config.SECONDARY_METRICS[key]

    def _apply_filters(df_):
        df_ = eval_condition(df_, condition, df_)
        try:
            res = f(df_[col])
        except TypeError:
            res = 0.0
        return res
    return _apply_filters(df)


def parse_primary(df, key):
    """Парсинг конфига показателей из PRIMARY_METRICS"""
    x = config.PRIMARY_METRICS[key]['options']['x']
    y = config.PRIMARY_METRICS[key]['options']['y']

    if len(x) < 3: x.append(None)
    if len(y) < 3: y.append(None)
    x_col, x_f, x_condition = x
    y_col, y_f, y_condition = y

    def _apply_filters(df_):
        df_ = eval_condition(df_, config.PRIMARY_METRICS[key]['filter_to'], df_)
        x_series = eval_condition(df_[x_col], x_condition, df_)
        y_series = eval_condition(df_[y_col], y_condition, df_)
        try:
            exprs = x_f(x_series) / y_f(y_series)
        except Exception:
            exprs = 0.0
        return exprs
    return _apply_filters(df)


def agg_totals(df, created_at, surcharge_id, split_pair):
    """
    Парсер словарей с метриками эксперимента

    Парсит PRIMARY_METRICS и SECONDARY_METRICS

    Нужна для функции get_totals()

    Parameters
    ----------
    df
    created_at
    surcharge_id
    split_pair

    Returns
    -------
    out : pd.Series
        Словарь метрик

    """
    names = {
        'surcharge_id': surcharge_id,
        'created_at': created_at,
        'split_pair': '_'.join(str(int(i)) for i in split_pair)
    }

    for key in config.SECONDARY_METRICS:
        names[key] = parse_secondary(df, key)

    for key in config.PRIMARY_METRICS:
        names[key] = parse_primary(df, key)

    return pd.Series(names)


def to_buckets(df: pd.DataFrame, bucket_n: int):
    """
    Бакетизация метрики

    Parameters
    ----------
    df
    bucket_n : int
        Количество бакетов. Регулируется в config.py

    Returns
    -------
    out : list
        Оптимальное колиество бакетов для метрики
    """
    return [i % bucket_n for i in df.index.values]


def pivot_with_lifts(df: pd.DataFrame, split_col):
    """
    Пивот результата agg_totals

    Parameters
    ----------
    df

    Returns
    -------

    """
    df = pd.melt(df, id_vars=['created_at', split_col, 'split_pair', 'surcharge_id']) \
        .pivot_table(index=['created_at', 'surcharge_id', 'split_pair', 'variable'], columns=split_col, values='value') \
        .reset_index()
    df.columns = ['created_at', 'surcharge_id', 'split_pair', 'metric', 'split_0', 'split_1']
    df['lift'] = (df['split_1'] - df['split_0']) / df['split_0']
    df.loc[np.isinf(df['lift']), 'lift'] = 0.0
    return df


def get_totals(df: pd.DataFrame, sur_id: int, split_pair: str, split_col: str):
    """
    Агрегация результатов эксперимента

    Parameters
    ----------
    df
    sur_id
    split_pair

    Returns
    -------
    out : pd.DataFrame
        Таблица с метриками (количество водителей, gmv, поездок и т.п.)
    """
    created_at = datetime.now().isoformat()
    df = df.groupby(split_col).apply(agg_totals, created_at=created_at, surcharge_id=sur_id, split_pair=split_pair).reset_index()
    df = pivot_with_lifts(df, split_col)
    return df


def upload_to_db(df: pd.DataFrame, sur_id: int):
    """
    Залив итоговых результатов эксперимента в segmentation

    Parameters
    ----------
    df
    sur_id

    Returns
    -------

    """
    try:
        df = df[['created_at','surcharge_id','split_pair','metric','split_0','split_1','lift','pvalue','power','is_significant','pvalue_adj','is_significant_adj']]
        mysql_segmentation = db.get_mysql_conn_segm()
        mysql_segmentation.insert_data_frame(f"surcharge_experiment_stats", df)
        logging.info(f'Results uploaded')
    except Exception as e:
        logging.error(e)
        logging.error(f'Cant upload data to results. surcharge_id: {sur_id}')
    return


def combine_splits(df: pd.DataFrame, group_col: str):
    return list(itertools.combinations(df.sort_values(by=group_col).split.dropna().unique(), 2))


if __name__ == '__main__':
    print("utils loaded")