import pandas as pd
import numpy as np
import config
import itertools
import math
import db
import query as q
import logging
from datetime import datetime, timedelta


def eval_condition(df: pd.DataFrame, condition=None):
    if condition is not None:
        list_condition = []
        for i in range(len(condition)):
            list_condition.append('(df.' + condition[i] + ')')
        list_condition = ' & '.join(list_condition)
        df = eval('df[(' + list_condition + ')]')
    else:
        pass
    return df


def get_raw_data(
        ci_id: int,
        dt_start: str,
        dt_end: str
):
    """
    Получение сырых данных

    Parameters
    ----------
    ci_id
    dt_start
    dt_end

    Returns
    -------
    df
    """
    dt_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if (type(dt_end) != str) or (dt_now < dt_end):
        dt_end = dt_now
    try:
        df = db.get_ch_main_conn().fetch(q.q_1(ci_id=ci_id, dt_start=dt_start, dt_end=dt_end))
        polygons = db.get_mysql_conn_prod().fetch(q.q_6())
        client_impacts = db.get_mysql_conn_db23().fetch(q.q_7())
        googly_locality = db.get_mysql_conn_prod().fetch(q.q_3())
        tariff_zones = db.get_mysql_conn_prod().fetch(q.q_4())
        tariff_groups = db.get_mysql_conn_prod().fetch(q.q_2())

        df = df.merge(polygons, on="polygon_from") \
            .merge(client_impacts, on="ci_id") \
            .merge(googly_locality, on="id_locality") \
            .merge(tariff_zones, on="id_tariff_zone") \
            .merge(tariff_groups, on='id_tariff_group')
        return df
    except Exception as e:
        print(e)
        return None


def to_buckets(df: pd.DataFrame, bucket_n: int):
    """
    Создание бакетов

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


def as_buckets(df: pd.DataFrame, bucket_n: int):
    """
    Создание бакетов

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
        "bucket": [i % bucket_n for i in range(0,len(users_distinct))]
    })
    df = df.merge(df_, on=config.USER_ID_COL)
    return df


def compress(
        df: pd.DataFrame,
        to_compress: int,
        metric: str,
        agg: dict = None,
        estimator=False,
        filter_to=None,
        split_col: str = config.SPLIT_COL):
    """
    Пайп компрессии с помощью бакетизации

    Parameters
    ----------
    df
    split_col : str
        Колонка со сплитами
    to_compress : bool
        Флаг необходимости компрессии
    agg : dict
        Предварительная агрегация перед присвоением бакета

    Returns
    -------
    out : pd.DataFrame
        Сжатый датафрейм с сохранением среднего

    """
    if filter_to is not None:
        df = eval_condition(df, filter_to)

    if to_compress == 1 or to_compress == True:
        bucket_n = min(len(df) / 2, config.BUCKET_N)
        if len(df) < 1000:
            bucket_n = min(len(df) / 2, config.MIN_BUCKET_N)
        df = as_buckets(df, bucket_n)
        df = df.groupby([split_col, 'bucket']).apply(agg).reset_index()
        # if estimator == 'ttest_ratio':
        #     return df

    return df


def agg_totals(x, created_at, ci_id, split_pair):
    """
    Parameters
    ----------
    x
    created_at
    ci_id

    Returns
    -------
    out : pd.Series
        Посчитанные метрики (количество водителей, gmv, поездок и т.п.)

    """
    names = {
        'ci_id': ci_id,
        'created_at': created_at,
        'split_pair': '_'.join(str(i) for i in split_pair),
        'clients': x['id_client'].nunique(),
        'views': x['views'].sum(),
        'orders': x['orders'].sum(),
        'trips': x['rides'].sum(),
        'costs': x['sale_part_ride'].sum(),
        'avg_client_bill': x['client_bill'].sum() / x['orders'].sum(),
        'tpс': x['rides'].sum() / x['id_client'].nunique(),
        'o2r': x['rides'].sum() / x['orders'].sum(),
        'v2r': x['rides'].sum() / x['views'].sum(),
        'v2o': x['orders'].sum() / x['views'].sum(),
        'cpt': x['sale_part_ride'].sum() / x['rides'].sum()
    }
    return pd.Series(names)


def pivot_with_lifts(df: pd.DataFrame, split_col):
    """
    Пивот результата agg_totals

    Parameters
    ----------
    df

    Returns
    -------

    """
    df = pd.melt(df, id_vars=['created_at', split_col, 'split_pair', 'ci_id']) \
        .pivot_table(index=['created_at', 'ci_id', 'split_pair', 'variable'], columns=split_col, values='value') \
        .reset_index()
    df.columns = ['created_at', 'ci_id', 'split_pair', 'metric', 'split_0', 'split_1']
    df['lift'] = (df['split_1'] - df['split_0']) / df['split_0']
    df.loc[np.isinf(df['lift']), 'lift'] = 0.0
    return df


def get_totals(df: pd.DataFrame, ci_id: int, split_pair: str, split_col: str):
    """
    Агрегация результатов эксперимента

    Parameters
    ----------
    df
    ci_id
    split_pair
    split_col

    Returns
    -------
    out : pd.DataFrame
        Таблица с метриками (количество водителей, gmv, поездок и т.п.)
    """
    created_at = datetime.now().isoformat()
    df = df.groupby(split_col).apply(agg_totals, created_at=created_at, ci_id=ci_id, split_pair=split_pair).reset_index()
    df = pivot_with_lifts(df, split_col)
    return df


def upload_to_db(df: pd.DataFrame, ci_id: int):
    """
    Залив итоговых результатов эксперимента в segmentation

    Parameters
    ----------
    df
    ci_id

    Returns
    -------

    """
    try:
        mysql_segmentation = db.get_mysql_conn_segm()
        mysql_segmentation.insert_data_frame(f"ci_momental_experiment_stats", df)
        logging.info(f'Results uploaded')
    except Exception as e:
        logging.error(e)
        logging.error(f'Cant upload data to results. ci_id: {ci_id}')
    return


def combine_splits(df: pd.DataFrame, split_col):
    return list(itertools.combinations(df.sort_values(by=split_col)[split_col].dropna().unique(), 2))


def replace_nan(df: pd.DataFrame):
    cols = ['pvalue', 'power', 'is_significant', 'pvalue_adj', 'is_significant_adj']
    for col in cols:
        df[col] = df[col].replace(np.nan, None)
    return df


if __name__ == '__main__':
    print("utils loaded")
