from scipy.stats import norm, t
import ast
import typing as T
import numpy as np
import pandas as pd
from statsmodels.stats.power import tt_ind_solve_power
from shapely.geometry import Point, Polygon, shape
from pyproj import Transformer
import constants as const

def time_frmt(time):
    if time < 10:
        time = '0' + str(int(time))
    return str(time)


def upload_results(df, df_name, conn):
    cols = "`,`".join([str(i) for i in df.columns.tolist()])
    cursor = conn.cursor()
    for i, row in df.iterrows():
        sql = "INSERT INTO `geominimal_experiment_" + df_name + "` (`" + cols + "`) VALUES (" + "%s," * (
                    len(row) - 1) + "%s)"
        cursor.execute(sql, tuple(row))
        conn.commit()


def add_point_driver(df: pd.DataFrame):
    point_driver = []
    for p in range(len(df)):
        point_driver.append(Point(float(df['driver_lat'][p]), float(df['driver_lon'][p])))
    return point_driver


def literal_to_tuple(col: pd.Series):
    return ast.literal_eval(col.replace('\"', '').replace('[', '(').replace('((', '[(').replace(']', ')').replace('))', ')]'))


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
        polygon_area = shape(coords).area
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
            if is_in == True:
                list_points.append(True)
                break
            elif coord == len(polygons) - 1:
                list_points.append(False)
    return list_points


def eval_condition(df: pd.DataFrame, col: pd.Series, condition=None):
    if condition is not None:
        list_condition = []
        for i in range(len(condition)):
            list_condition.append('(df.' + condition[i] + ')')
        list_condition = ' & '.join(list_condition)
        col = eval('col[(' + list_condition + ')]')
        df = eval('df[(' + list_condition + ')]')
    else:
        pass
    return col, df


def check_zero(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0

def agg_totals(x, updated_at: str, campaign_id: str):
    # x = x[(x['retro'] == 'tw')]
    tpgd = check_zero(x['order_id'][(x['is_cp'] == True) & (x['retro'] == 'tw')].nunique(), x['id_driver'][
        (x['is_geominimal'] == True) & (x['is_ar'] == True) & (x['retro'] == 'tw')].nunique())
    rr = check_zero(
        x['id_driver'][(x['is_geominimal'] == True) & (x['is_ar'] == True) & (x['retro'] == 'tw')].nunique(),
        x['id_driver'][(x['retro'] == 'tw')].nunique())
    ar = check_zero(x['order_id'][(x['is_ar'] == True) & (x['retro'] == 'tw')].nunique(),
                    x['order_id'][(x['retro'] == 'tw')].nunique())
    cpt = check_zero(x['damage'][(x['is_cp'] == True) & (x['retro'] == 'tw')].sum(),
                     x['order_id'][(x['is_cp'] == True) & (x['retro'] == 'tw')].nunique())
    o2r_cnbl = check_zero(x['order_id'][(x['is_driver_in_cnbl_poly'] == True) & (x['is_cp'] == True) & (x['retro'] == 'tw')].nunique(),
                     x['order_id'][(x['is_driver_in_cnbl_poly'] == True) & (x['retro'] == 'tw')].nunique())
    of2r_cnbl = check_zero(x['order_id'][((x['is_driver_in_cnbl_poly'] == True) & x['is_cp'] == True) & (x['retro'] == 'tw')].nunique(),
                x['fss_id'][(x['is_driver_in_cnbl_poly'] == True) & (x['retro'] == 'tw')].nunique())
    o2r = check_zero(x['order_id'][(x['is_cp'] == True) & (x['retro'] == 'tw')].nunique(),
               x['order_id'][(x['retro'] == 'tw')].nunique())
    of2r = check_zero(x['order_id'][(x['is_cp'] == True) & (x['retro'] == 'tw')].nunique(),
                x['fss_id'][(x['retro'] == 'tw')].nunique())
    sh_per_driver = check_zero((x['supply_sec'][(x['retro'] == 'tw')].sum() / 3600),  x['id_driver'][
            (x['retro'] == 'tw')].nunique())
    sh = check_zero(x['supply_sec'][(x['retro'] == 'tw')].sum(), 3600)
    names = {
        'updated_at': updated_at,
        'campaign_id': campaign_id,
        'drivers': x['id_driver'][(x['retro'] == 'tw')].nunique(),
        'drivers_is_geo': x['id_driver'][
            (x['is_geominimal'] == True) & (x['is_ar'] == True) & (x['retro'] == 'tw')].nunique(),
        'drivers_is_cnbl': x['id_driver'][
            (x['is_driver_in_cnbl_poly'] == True) & (x['is_ar'] == True) & (x['retro'] == 'tw')].nunique(),
        'suggests_is_cnbl': x['fss_id'][(x['is_driver_in_cnbl_poly'] == True) & (x['retro'] == 'tw')].nunique(),
        'orders_is_cnbl': x['order_id'][(x['is_driver_in_cnbl_poly'] == True) & (x['retro'] == 'tw')].nunique(),
        'trips_is_cnbl': x['order_id'][(x['is_driver_in_cnbl_poly'] == True) & (x['is_cp'] == True) & (x['retro'] == 'tw')].nunique(),
        'o2r_cnbl': o2r_cnbl,
        'o2r': o2r,
        'of2r': of2r,
        'of2r_cnbl': of2r_cnbl,
        'orders': x['order_id'][(x['retro'] == 'tw')].nunique(),
        'accepted_orders': x['order_id'][(x['is_ar'] == True) & (x['retro'] == 'tw')].nunique(),
        'trips': x['order_id'][(x['is_cp'] == True) & (x['retro'] == 'tw')].nunique(),
        'suggests': x['fss_id'][(x['retro'] == 'tw')].nunique(),
        'ar': ar,
        'tpgd': tpgd,
        'sh': sh,
        'sh_per_driver': sh_per_driver,
        'rr': rr,
        'cost': x['damage'][(x['is_cp'] == True) & (x['retro'] == 'tw')].sum(),
        'cpt': cpt
    }
    return pd.Series(names)

def get_mde(n0: int, n1: int, sd0: float, sd1: float, y_c: float, alpha=const.CONFIDENCE_LEVEL, pwr=const.POWER_LEVEL):
    d_f = n0 + n1 - 2
    S = np.sqrt((sd1 ** 2 / n1) + (sd0 ** 2 / n0))
    M = t.ppf(pwr, d_f) - t.ppf(alpha / 2, d_f)
    mde = M * S / y_c
    return mde


def bootstrap_ratio_test(
        data: pd.DataFrame,
        x: str,
        y: str,
        split: str,
        x_f: T.Callable,
        y_f: T.Callable,
        user_level_col: str,
        x_condition=None,
        y_condition=None,
        metric=None,
        boot_it=1000,
        conf_level=0.95,
        bias_correction=False):
    data_splitted = [x for _, x in data.groupby(split)]
    boot_data = {'lift': []}
    split0 = data_splitted[0]
    split1 = data_splitted[1]
    orig_y0 = y_f(eval_condition(split0, split0[y], y_condition)[0])
    orig_y1 = y_f(eval_condition(split1, split1[y], y_condition)[0])
    orig_x0 = x_f(eval_condition(split0, split0[x], x_condition)[0])
    orig_x1 = x_f(eval_condition(split1, split1[x], x_condition)[0])
    orig_mean = check_zero(orig_x1, orig_y1) - check_zero(orig_x0, orig_y0)

    for i in range(boot_it):
        s0 = split0[
            split0[user_level_col].isin(split0[user_level_col].sample(split0[user_level_col].nunique(), replace=True))]
        s1 = split1[
            split1[user_level_col].isin(split1[user_level_col].sample(split1[user_level_col].nunique(), replace=True))]
        y0 = y_f(eval_condition(s0, s0[y], y_condition)[0])
        y1 = y_f(eval_condition(s1, s1[y], y_condition)[0])
        x0 = x_f(eval_condition(s0, s0[x], x_condition)[0])
        x1 = x_f(eval_condition(s1, s1[x], x_condition)[0])
        if (y0 == 0 or y1 == 0) or (x0 == 0 or x1 == 0):
            return metric, None, False
        s0_ratio = x0 / y0
        s1_ratio = x1 / y1
        boot_data['lift'].append(s1_ratio - s0_ratio)

    boot_mean = np.mean(boot_data['lift'])
    delta_val = orig_mean - boot_mean
    if bias_correction:
        boot_data['lift'] = [i + delta_val for i in boot_data['lift']]
    n0 = data_splitted[0]['id_driver'].nunique()
    p1 = norm.cdf(x=0, loc=np.mean(boot_data['lift']), scale=np.std(boot_data['lift']))
    p2 = norm.cdf(x=0, loc=-np.mean(boot_data['lift']), scale=np.std(boot_data['lift']))
    pvalue = min(p1, p2) * 2
    mark = (pvalue < 1 - conf_level)
    mde = tt_ind_solve_power(alpha=const.CONFIDENCE_LEVEL, power=const.POWER_LEVEL, nobs1=n0, ratio=1)

    return metric, pvalue, mark, mde