import pandas as pd
import numpy as np
import logging
import itertools
import logging
from pandasql import sqldf
import datetime
import ast
import json
from shapely.geometry import Point, Polygon, shape, MultiPolygon, LineString
from shapely.ops import cascaded_union
from pyproj import Transformer


# Собственные модули
import config
import query as q
import db

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

def time_frmt(time):
    if time < 10:
        time = '0' + str(int(time))
        return str(time)
    return str(int(time))


def add_point_driver(df: pd.DataFrame):
    point_driver = []
    for p in range(len(df)):
        point_driver.append(Point(float(df['driver_lat'][p]), float(df['driver_lon'][p])))
    return point_driver


def literal_to_tuple(col: pd.Series):
    return col
    # return ast.literal_eval(col.replace('\"', '').replace('[', '(').replace('((', '[(').replace(']', ')').replace('))', ')]'))


def to_polygon(polygon_col: pd.Series):
    polygons = []
    for poly in range(len(polygon_col)):
        polygons.append(Polygon(literal_to_tuple(polygon_col[poly])))
    return polygons


def transform_coordinates(coordinates, in_proj='EPSG:4326', out_proj='EPSG:3857'):
    coord_tuple = []
    for (x,y) in coordinates:
        transformer = Transformer.from_crs(in_proj, out_proj)
        lat, lon = transformer.transform(x,y)
        coord_tuple.append((lat, lon))
    return coord_tuple


def polygon_to_list(polygon):
    return list(zip(*polygon.exterior.coords.xy))


def add_cannibalization_area(polygon_col: pd.Series, added_area: int):
    buffered_areas = []
    cannibalization_areas = []
    for coord in polygon_col:
        poly = literal_to_tuple(coord)
        coords = Polygon(transform_coordinates(poly))
        # polygon_area = shape(coords).area
        added_area = added_area
        buffer_area = coords.buffer(added_area)
        buffer_area = polygon_to_list(buffer_area)
        buffer_area = transform_coordinates(buffer_area, in_proj='EPSG:3857', out_proj='EPSG:4326')
        original_area = polygon_to_list(coords)
        original_area = transform_coordinates(original_area, in_proj='EPSG:3857', out_proj='EPSG:4326')
        area_diff = Polygon(original_area).buffer(0).symmetric_difference(Polygon(buffer_area).buffer(0))
        buffered_areas.append(Polygon(buffer_area))
        cannibalization_areas.append(area_diff)
    return buffered_areas, cannibalization_areas


def add_driver_location(points: pd.DataFrame, polygons: pd.DataFrame, coords_col: str):
    list_points = []
    for point in range(len(points)):
        for coord in range(len(polygons)):
            is_in = points['point_driver'][point].intersects(polygons[coords_col][coord])
            if is_in:
                list_points.append(True)
                break
            elif coord == len(polygons) - 1:
                list_points.append(False)
    return list_points


def check_zero(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0




def preprocess_gch(df, campaign_id):
    """
    Конвертация таймингов из расписания в условия для эксзасола

    :param df:
    :param campaign_id:
    :return:
    """
    df = df[(df['campaign_id'] == campaign_id) & (~np.isnan(df['week_day']))].reset_index()
    timings = []
    for time in range(len(df)):
        from_date = df['from_date'][time]
        from_hh = df['from_hour'][time]
        if from_hh == 24:
            from_hh = 23
        from_mm = df['from_minute'][time]

        # until_date = datetime.today()
        until_date = df['until_date'][time]
        until_hh = df['until_hour'][time]
        if until_hh == 24:
            until_hh = 23

        until_mm = df['until_minute'][time]

        wd = df['week_day'][time]

        timings.append(
            f"(HOUR(ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET)) BETWEEN HOUR('0001-01-01 {int(from_hh)}:{int(from_mm)}') \
                AND HOUR('0001-01-01 {int(until_hh)}:{int(until_mm)}')\
                AND TO_CHAR(ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET),'D') = {wd})")
    if len(timings) == 0:
        timings = "1=1"
    else:
        timings = "\n OR ".join(timings)

    # процессинг gch
    geominimal_settings = pd.DataFrame(
        columns=['id_locality', 'campaign_id', 'week_day', 'from_date', 'until_date', 'from_hour', 'from_minute', 'until_hour',
                 'until_minute',
                 'poly', 'coords'])

    for r in df.itertuples():

        list_poly = []

        for p in json.loads(r.polygons):
            coords_list = []
            for c in p.get('coords'):
                coords_list.append(tuple([float(c[1]), float(c[0])]))
            list_poly.append(Polygon(coords_list))

            # если в данном временном интервале некоторые полигоны пересекаются, поправляем через cascaded_union
            cu = [cascaded_union(list_poly)] if type(
                cascaded_union(list_poly)) == Polygon else cascaded_union(list_poly)
            mp = MultiPolygon(cu)

            geominimal_settings = geominimal_settings.append({
                'id_locality': r.id_locality, 'campaign_id': r.campaign_id, 'from_date': r.from_date, 'until_date': r.until_date,
                'week_day': int(r.week_day), 'from_hour': r.from_hour, 'from_minute': r.from_minute,
                'until_hour': r.until_hour, 'until_minute': r.until_minute, 'poly': mp, 'coords': coords_list
            }, ignore_index=True)

    return timings, from_date, until_date, geominimal_settings


def preprocess(suggest_df, gch):
    print("start processing...")
    suggest_df['is_ar'] = suggest_df['action'].isin([1, -2])
    suggest_df['driver_picked_up_order'] = (suggest_df['id_driver'] == suggest_df['driver'])
    suggest_df['is_cp'] = suggest_df['status'].isin(['CP'])
    suggest_df['is_geominimal_trip'] = suggest_df['campaign_id'].notna()
    gm_drivers = suggest_df.groupby('id_driver')['is_geominimal_trip'].agg('max').reset_index()
    gm_drivers = gm_drivers.rename({'is_geominimal_trip': 'is_geominimal_driver'}, axis='columns')
    suggest_df = suggest_df.merge(gm_drivers, how="left", on='id_driver')

    #gch['cnbl_coords'] = add_cannibalization_area(gch['coords'], config.BUFFER_AREA)[1]
    #print("add_cannibalization_area done")
    # парсинг позиции водителя на момент получения саджеста
    suggest_df['point_driver'] = add_point_driver(suggest_df)
    print("point_driver done")
    #suggest_df['is_driver_in_cnbl_poly'] = add_driver_location(suggest_df, gch, 'cnbl_coords')
    #print("add_driver_location done")
    suggest_df = suggest_df.sort_values(by=['split'], ascending=[True])

    return suggest_df, gch



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
        campaign_id: int,
        locality,
        timings,
        from_date,
        until_date,
        gch,
        geomstat_df,
        driver_splits_df
):
    """
    Получение сырых данных


    """
    dt_now = datetime.datetime.now().date()
    if (dt_now < until_date):
        until_date = dt_now

    for c in ['from', 'until']:
        gch[c] = gch.apply(
            lambda x: datetime.datetime(2000, 1, 1) + datetime.timedelta(hours=x[f'{c}_hour'])
                      + datetime.timedelta(minutes=x[f'{c}_minute']), axis=1)

    sql_tw = q.q_5_new_format(locality=locality,timings=timings,
                   from_date=from_date,until_date=until_date,period='tw')
    sql_lw = q.q_5_new_format(locality=locality, timings=timings,
                   from_date=from_date, until_date=until_date, period='lw')
    sql = f"""({sql_tw}) UNION ALL ({sql_lw})"""
    print(sql)
    suggest_df = pd.DataFrame(db.get_exasol_connection().execute(sql).fetchall())
    suggest_df.columns = map(str.lower, suggest_df.columns)
    suggest_df['suggest_num'] = suggest_df.sort_values(['date_suggest1'], ascending=False).groupby(["id", "id_driver"]).cumcount() + 1
    suggest_df = suggest_df[(suggest_df.suggest_num == 1)]

    suggest_df['week_day'] = suggest_df['week_day'].astype(int)

    suggest_df['hhmm'] = suggest_df.apply(lambda x: datetime.datetime(2000, 1, 1) + datetime.timedelta(hours=x['h'])
                                        + datetime.timedelta(minutes=x['m']), axis=1)
    suggest_df['point'] = suggest_df.apply(lambda x: Point(float(x.driver_lon), float(x.driver_lat)), axis=1)
    gch = gch[['week_day', 'from', 'until', 'poly', 'from_hour', 'from_minute', 'until_hour', 'until_minute']]
    suggest_df = gch.merge(suggest_df, how='left', on='week_day')
    suggest_df = suggest_df[(suggest_df['hhmm'] >= suggest_df['from']) & (suggest_df['hhmm'] < suggest_df['until'])]
    suggest_df['point_is_in'] = suggest_df.apply(lambda x: x.poly.contains(x.point), axis=1)
    suggest_df = suggest_df.drop(columns=['week_day', 'from', 'until', 'poly', 'hhmm', 'point'])
    suggest_df = suggest_df[suggest_df['point_is_in'] == 1]


    timings = []
    for time in range(len(gch)):
        from_hh = int(float(time_frmt(gch['from_hour'][time])))
        from_mm = time_frmt(gch['from_minute'][time])
        from_s = '00'
        until_hh = int(float(time_frmt(gch['until_hour'][time])))
        until_mm = time_frmt(gch['until_minute'][time])
        until_s = '00'
        wd = gch['week_day'][time]
        if wd == 7:
            wd = 0
        if from_hh < 10:
            from_hh = "0" + str(from_hh)
        if until_hh < 10:
            until_hh = "0" + str(until_hh)
        if until_hh == 24:
            until_hh = 23
            until_mm = 59
            until_s = '59'

        timings.append(
            f"(strftime('%H:%M:%S', date_suggest1) BETWEEN strftime('%H:%M:%S', \'{from_hh}:{from_mm}:{from_s}\') AND strftime('%H:%M:%S', \'{until_hh}:{until_mm}:{until_s}\') AND strftime('%w', date_suggest1) = \'{int(wd)}\')")


    if len(timings) == 0:
        timings = "1=1"
    else:
        timings = "\n OR ".join(timings)
    query_locals = f'''SELECT * FROM suggest_df WHERE 1 =1 AND ({timings})'''
    suggest_df = sqldf(query_locals, locals())
    suggest_df = suggest_df.rename(columns={'id': 'order_id'})
    # финальная слепка
    suggest_df = suggest_df.merge(geomstat_df, how='left', on='order_id') \
        .merge(driver_splits_df, how='left', left_on='id_driver', right_on='id')

    return suggest_df



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
    if len(df) < 1000:
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


def agg_totals(df, created_at, campaign_id, split_pair):
    """
    Парсер словарей с метриками эксперимента

    Парсит PRIMARY_METRICS и SECONDARY_METRICS

    Нужна для функции get_totals()

    Parameters
    ----------
    df
    created_at
    campaign_id
    split_pair

    Returns
    -------
    out : pd.Series
        Словарь метрик

    """
    names = {
        'campaign_id': campaign_id,
        'updated_at': created_at,
        'split_pair': '_'.join(str(int(i)) for i in split_pair)
    }

    for key in config.SECONDARY_METRICS:
        names[key] = parse_secondary(df, key)

    for key in config.PRIMARY_METRICS:
        names[key] = parse_primary(df, key)

    return pd.Series(names)


def pivot_with_lifts(df: pd.DataFrame, split_col):
    """
    Пивот результата agg_totals

    Необходимо для джойна результатов расчета стат. значимости для комбинаций пар вариантов эксперимента

    Parameters
    ----------
    df

    Returns
    -------

    """
    df = pd.melt(df, id_vars=['updated_at', split_col, 'split_pair', 'campaign_id']) \
        .pivot_table(index=['updated_at', 'campaign_id', 'split_pair', 'variable'], columns=split_col, values='value') \
        .reset_index()
    df.columns = ['updated_at', 'campaign_id', 'split_pair', 'metric', 'split_0', 'split_1']
    df['lift'] = (df['split_1'] - df['split_0']) / df['split_0']
    df.loc[np.isinf(df['lift']), 'lift'] = 0.0
    return df


def get_totals(df: pd.DataFrame, campaign_id: int, split_pair: str, split_col: str):
    """
    Агрегация результатов эксперимента

    Parameters
    ----------
    df
    campaign_id
    split_pair
    split_col

    Returns
    -------
    out : pd.DataFrame
        Таблица с метриками
    """
    created_at = datetime.datetime.now().isoformat()
    df_pivoted = df.groupby(split_col).apply(agg_totals, created_at=created_at, campaign_id=campaign_id, split_pair=split_pair).reset_index()
    df = pivot_with_lifts(df_pivoted, split_col)
    return df, df_pivoted


def combine_splits(df: pd.DataFrame, split_col: str):
    """
    Получение комбинаций сплитов эксперимента
    Например, для [1,2,3] вернет [(1,2), (1,3), (2,3)]

    Если необходимо брать сплиты по правилам из конфига severx.public.spliter_configuration,
    то нужно поставить флаг в config.GET_SPLITS_FROM_DESCRIPTION = True

    Parameters
    ----------
    df : pd.Dataframe
    split_col : str
    rules pd.Series

    Returns
    -------

    """

    pairs = list(itertools.combinations(df.sort_values(by=split_col)[split_col].dropna().unique(), 2))

    return pairs


def upload_to_db(tables, campaign_id):
    """
    Залив данных в сегментацию
    :param tables:
    :return:
    """
    for table in tables:
        if table == 'stats':
            tables[table] = tables[table][['updated_at', 'campaign_id', 'split_pair', 'metric', 'split_0', 'split_1',
                           'lift', 'pvalue', 'is_significant', 'mde', 'pvalue_adj', 'is_significant_adj']]
        elif table == 'totals':
            tables[table][['of2r_cnbl', 'suggests_is_cnbl', 'orders_is_cnbl', 'o2r_cnbl', 'drivers_is_cnbl', 'trips_is_cnbl']] = np.nan
            tables[table] = tables[table][['split', 'updated_at', 'split_pair', 'campaign_id', 'drivers', 'drivers_is_geo',
                           'drivers_is_cnbl', 'orders', 'accepted_orders', 'suggests', 'ar', 'tpgd', 'sh', 'trips',
                           'sh_per_driver', 'rr', 'cost', 'cpt', 'trips_is_cnbl', 'suggests_is_cnbl', 'orders_is_cnbl',
                           'o2r_cnbl', 'o2r', 'of2r', 'of2r_cnbl', 'o2r_wow']]
        try:
            mysql_segmentation = db.get_mysql_conn_segm()
            mysql_segmentation.insert_data_frame(f"geominimal_experiment_{table}", tables[table])
            logging.info(f'Results uploaded')
        except Exception as e:
            logging.error(e)
            logging.error(f'Cant upload data to results. campaign_id: {campaign_id}')

if __name__ == '__main__':
    print("utils loaded")