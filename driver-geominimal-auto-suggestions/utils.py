__author__  = "Igor Singin"
__version__ = "2022-01-10"
__email__   = "i.singin@city-mobil.ru"
__status__  = "Prototype"

import gc
import sys
import pandahouse
import datetime
import time
import warnings
warnings.simplefilter('ignore')
from sklearn.cluster import KMeans
import h3
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import silhouette_samples, silhouette_score
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from geodaisy import GeoObject
import geopandas as gpd
from shapely import geometry as shapely_geometry
from configparser import ConfigParser


config = ConfigParser()
config.read("config.ini")

labels             = str(config['clustering']['labels'])
min_dbscan_samples = int(config['clustering']['min_dbscan_samples'])
print_info_flag    = bool(config['printing_logging']['print_info_flag'])

def add_lat_lon_cols(df, hexagon_column, logger):
    '''
    Добавляем в датасет данные о центрах гексагонов.
    Input:
    df             - датасет
    hexagon_column - название столбца с гексагонами
    Output:
    df             - датасет обогащённый координатами
    '''

    lat_list, long_list = [], []
    ### Проверяем сущестрование в датафрейсе данных с которыми будем работать
    if((hexagon_column in df.columns) & (df.shape[0] > 0)):
        try:
            for lat, long in df[hexagon_column].apply(h3.h3_to_geo):
                lat_list.append(lat)
                long_list.append(long)
            df['geo_lat'] = lat_list
            df['geo_long'] = long_list
        except Exception as err:
            logger.error(f"Unexpected error in add_lat_lon_cols {err}, {type(err)}")
            exit(1)
    else:
            logger.error('Error with training dataframe! Cant add lat/lon coordinates!')
            exit(1)
    return df

def get_filtered_data_daily_tr(df, groupby_list, quantile_views, quantile_d2v):
    '''
    Фильтрует гексагоны в локации и на основе квантилей по views и d2v отфильтровывает
    гексагоны которые нужно бустить геоминималками.
    Возвращает отфильтрованный датасет.
    '''

    d2v_threshold = df.groupby(groupby_list, as_index=False)['d2v'].quantile(quantile_d2v)
    views_threshold = df.groupby(groupby_list, as_index=False)['uviews_agg'].quantile(quantile_views)

    thresholds = d2v_threshold.merge(views_threshold, on=groupby_list)
    temp_df = df.merge(thresholds, on=groupby_list, suffixes=(None, '_tr'))

    result = temp_df[(temp_df['uviews_agg'] >= temp_df['uviews_agg_tr']) & \
                     (temp_df['d2v'] <= temp_df['d2v_tr'])
                     ]
    result = result.drop(['uviews_agg_tr', 'd2v_tr'], axis=1)

    return result


def join_2_datasets(df1, df2, index_col,logger):
    '''
    LEFT JOIN 2 датасетов по задаваемому ключу с использованием индексов(для ускорения)

    df1, - датасет слева, к кторому джойним
    df2, - датасет справа, который джойним
    index_col - столбец, по которому джойним
    '''

    df1.set_index(index_col, inplace=True)
    df2.set_index(index_col, inplace=True)

    temp = pd.merge(df1
                    , df2
                    , left_index=True
                    , right_index=True
                    , how='left'
                    , copy=False
                    )
    del df1, df2
    gc.collect()
    temp.reset_index(inplace=True)

    return temp

def find_num_clusters_silhouette(X_scale, logger):
    ### FIND NUMBER OF CLUSTERS WITH SILHOUETTE
    '''
    Находим количество кластеров методом силуетов (это не метод локтя!)
    Input:
    X_scale - датасет

    Output:
    number_of_clusters - количество кластеров
    '''
    km_silhouette = []
    ### Количство класетров - от 2 до 9
    clusters_num_list = range(2, min(9, X_scale.shape[0]))
    for i in clusters_num_list:
        km = KMeans(n_clusters=i, random_state=17).fit(X_scale)
        preds = km.predict(X_scale)
        silhouette = silhouette_score(X_scale, preds)
        km_silhouette.append(silhouette)

    number_of_clusters = int(list(clusters_num_list)[np.argmax(km_silhouette)])

    return number_of_clusters

def find_cnt_neighbours_in_cluster(hexagon, cluster, ring, logger):
    '''
    Функция считает количество соседей для гексагона из заданного кластера
    hexagon - гексагон для которого считаем кол-во соседей
    cluster - list гексагонов принадлежащих кластеру
    ring    - расстояние в гексагонах от заданного

    return  - int количество соседей
    '''

    if (len(cluster) < 1):
        logger.warning('Empty cluster in find_cnt_neighbours_in_cluster')
    if ((ring == 0) | (ring > 12)):
        logger.warning('Wrong ring size in find_cnt_neighbours_in_cluster, ring = {0}'.format(ring))

    # for hexagon in cluster:
    num_hexagon_neighb = 0
    for another_hexagon in cluster:
        if (another_hexagon in cluster) & (hexagon != another_hexagon):
            if (another_hexagon in list(h3.k_ring_distances(hexagon, ring)[ring])):
                num_hexagon_neighb = num_hexagon_neighb + 1
            else:
                num_hexagon_neighb = num_hexagon_neighb + 0

    return num_hexagon_neighb

def drop_single_hexagons(df, labels, logger, cnt_neighbors):
    '''
    Удаляем гексагон, который оказался в одиночестве, в отрыве от своего кластера.
    Удаляем гексагон, вокруг которого менее cnt_neighbors гексагонов из его кластера
    '''

    l_cnt_neighb = []

    ### Для каждой строчки в датасете считаем кол-во соседей из его кластера
    for i in range(df.shape[0]):
        hexagon = df.iloc[i, :]['hex_id']
        ### находим номер кластера
        labels_km_sil = df.iloc[i, :][labels]
        ### находим гексагоны в кластере
        cluster = list(df[df[labels] == labels_km_sil]['hex_id'])
        ### количество соседей
        cnt_neighb = find_cnt_neighbours_in_cluster(hexagon, cluster, 1 ,logger)
        l_cnt_neighb.append(cnt_neighb)

    ### Отфильтровываем
    df['cluster_neigbors'] = l_cnt_neighb
    df = df[df.cluster_neigbors > cnt_neighbors]
    df = df.drop('cluster_neigbors', axis=1)

    return df

def find_cnt_foreign_clusters_for_hexagon(hexagon, df, labels, ring, logger):
    '''
    Находим рядом с заданным гексагоном кластер с максимальным количеством гексагонов, соседствующих
    с заданным гексагоном. Ищем ситуации, когда один гексагон окружен другим кластером.

    hexagon - гексагон h3
    df      -
    ring    - дальность поиска
    '''

    ### находим кластер заданного гексагона
    hexagon_label = df[df['hex_id'] == hexagon][labels].unique()

    l_name_mager_neighb = []
    l_cnt_mager_neighb = []

    list_of_clusters = list(df[labels].unique())

    ### Если есть как минимум 2 кластера:
    if len(list_of_clusters) >= 2:
        ### для каждого кластера в локации
        for label in list_of_clusters:
            ### для каждого кластера не равного кластеру исходного гексагона
            if label not in hexagon_label:
                ### набор гексагонов из которых состоит кластер сосдствующий с гексагоном
                cluster = list(df[df[labels] == label]['hex_id'])
                ### количество гексагонов-соседей из этого клфстера
                cnt_neighb = find_cnt_neighbours_in_cluster(hexagon, cluster, ring,logger)

                l_name_mager_neighb.append(label)
                l_cnt_mager_neighb.append(cnt_neighb)

                ### кол-во соседей от одного кластера рядом с заданным гексагоном
        max_foreign_label = max(l_cnt_mager_neighb)
        index = l_cnt_mager_neighb.index(max_foreign_label)
        ### имя кластера имеющего максимум соседей с заданным гексагоном
        name_max_foreign_label = l_name_mager_neighb[index]
    ### Если у нас был только 1 кластер
    else:
        max_foreign_label = -1
        name_max_foreign_label = -1

    return name_max_foreign_label, max_foreign_label

def reassign_hexagon_by_neighbors(df, labels, logger, cnt_neighbors):
    '''
    Переназначаем кластер гексагона на основе кластеров, которые его окружают. Если гексагон
    окружают более cnt_neighbors гексагонов, принадлежащих какомо-то одному кластеру, он
    меняет свой кластер на кластер соседей.
    '''
    for hexagon in df['hex_id'].unique():
        ### находим кластер заданного гексагона
        hexagon_label = df[df['hex_id'] == hexagon][labels].unique()
        ### Находим каким кластером окружен гексагон
        name_max_foreign_label, max_foreign_label = find_cnt_foreign_clusters_for_hexagon(hexagon, df, labels, 1,logger)

        if (max_foreign_label > cnt_neighbors) & (name_max_foreign_label != hexagon_label):
            df.loc[df['hex_id'] == hexagon, labels] = name_max_foreign_label

    return df

def dist_2_hexagons(gexagon1, gexagon2, logger):
    '''
    Находим расстояние между центрами 2 гексагонов

    In: gexagon1, gexagon2
    Out: distance (not km!)
    '''

    lat0, lon0 = h3.h3_to_geo(gexagon1)
    lat1, lon1 = h3.h3_to_geo(gexagon2)

    distance = np.sqrt((lat1 - lat0) ** 2 + (lon1 - lon0) ** 2)

    return distance


def dist_round_gexagon(gexagon, logger):
    '''
    Находим минимальное, среднее, максимальное расстояние от центра гексагона
    до центра соседних гексагонов. Эти расстояния в зависимости от направления различаются!
    '''

    l_dist = []
    for i in range(6):
        n0_gex = list(h3.k_ring_distances(gexagon, 1)[1])[i]
        dist = dist_2_hexagons(gexagon, n0_gex,logger)
        l_dist.append(dist)

    return np.max(l_dist)


def calc_max_dbscan_dist(df, logger):
    '''
    Для массива гексагонов рассчитываем среднее/максимальное расстояние до центра соседнего гексагона
    '''

    if (df.shape[0] < 3):
        logger.warning('To small dataframe size in calc_max_dbscan_dist = {0} gexagons'.format(df.shape[0]))

    l_max = []
    for gexagon in list(df['hex_id'].unique()):
        max_dist = dist_round_gexagon(gexagon,logger)

        l_max.append(max_dist)

    return (0.2 * np.mean(l_max) + 0.8 * np.max(l_max))

def fill_hole_in_cluster(df, data, logger, cnt_neighb_treshold):
    '''
    Функция заполняет "дырку" в кластере, присваивая ей индекс кластера, окружающего её.
    Если вокруг гексагона > cnt_neighb_treshold соседей из одного кластера, а сам он
    не принадлежит никакому кластеру, то он получает индекс этого кластера

    df - фильтрованный датасет
    data - нефильтрованный датасет
    cnt_neighb_treshold - количество соседних гексагонов

    return df
    '''

    for label in list(df['labels_geoclusters'].unique()):
        df_temp = df[df['labels_geoclusters'] == label]

        ### Находим геометрический центр кластера
        cluster_center_lat = (df_temp['geo_lat'].min() + df_temp['geo_lat'].max()) / 2
        cluster_center_long = (df_temp['geo_long'].min() + df_temp['geo_long'].max()) / 2

        ### Находим центральный гексагон в кластере
        center_gexagon_of_cluster = h3.geo_to_h3(cluster_center_lat, cluster_center_long, 8)

        border_gexagon_of_cluster1 = h3.geo_to_h3(df_temp['geo_lat'].min(), df_temp['geo_long'].min(), 8)
        border_gexagon_of_cluster2 = h3.geo_to_h3(df_temp['geo_lat'].max(), df_temp['geo_long'].max(), 8)

        ### Находим максимальный радиус поиска (в количестве гексагонов)
        search_radius_gexagons = max(h3.h3_distance(center_gexagon_of_cluster, border_gexagon_of_cluster1)
                                     , h3.h3_distance(center_gexagon_of_cluster, border_gexagon_of_cluster2)
                                     )
        search_radius_gexagons = min(search_radius_gexagons + 2, 20)
        list_of_clusters = list(df_temp[labels].unique())

        ### Если центральный гексагон пустой, то берём первый попавшийся гексагон из первого попавшегося кластера
        ### !Только для вычисления  locality_id, weekday, houry
        if (center_gexagon_of_cluster not in df_temp['hex_id'].unique()):
            base_gexagon = df_temp[df_temp['labels_geoclusters'] == list_of_clusters[0]]['hex_id'].unique()[0]
        else:
            base_gexagon = center_gexagon_of_cluster

        locality = df_temp[df_temp['hex_id'] == base_gexagon]['main_id_locality'].unique()[0]
        weekday = df_temp[df_temp['hex_id'] == base_gexagon]['weekday'].unique()[0]
        houry = df_temp[df_temp['hex_id'] == base_gexagon]['hour'].unique()[0]

        ### Перебираем радиусы
        for radius in range(0, search_radius_gexagons + 1):
            ### Список гексагонов которые будем проверять
            hexagons = list(h3.k_ring_distances(center_gexagon_of_cluster, radius)[radius])
            ### Для каждого гексагона внутри радиуса
            for hexagon in hexagons:
                for label in list_of_clusters:
                    ### набор гексагонов из которых состоит кластер сосдствующий с гексагоном
                    cluster = list(df_temp[df_temp[labels] == label]['hex_id'])
                    ### количество гексагонов-соседей из этого кластера
                    cnt_neighb = find_cnt_neighbours_in_cluster(hexagon, cluster, 1,logger)
                    if (cnt_neighb > cnt_neighb_treshold) & (hexagon not in cluster):
                        #### вытаскиваем строку данных по гексагону и времени из основного датасета
                        #### и добавляем её
                        temp_df = data[(data['hex_id'] == hexagon) & \
                                       (data['weekday'] == weekday) & \
                                       (data['hour'] == houry) & \
                                       (data['main_id_locality'] == locality)
                                       ]
                        temp_df['labels_geoclusters'] = label
                        temp_df['cluster_neigbors'] = cnt_neighb

                        df = df.append(temp_df)
    return df

def find_geoclusters(df, labels, min_dbscan_samples, logger):
    '''
    Находим геокластеры в городе, и раскрашиваем столбец labels в эти кластеры. Кластеры размечаются DBScan
    исключительно на основе координат
    '''
    if (df.shape[0] < 3):
        logger.warning('To small dataframe size for finding geoclusters = {0} gexagons'.format(df.shape[0]))

    ## Находим максимальную дистанцию между центрами гексагонов
    ### для последующего деления на геокластеры DBScan
    max_dist_betweeen_gexagons = calc_max_dbscan_dist(df,logger)

    # X = df[[ 'geo_lat' ,'geo_long']].fillna(0)
    X = df[['geo_lat', 'geo_long']].dropna()
    ### Find GEOCLUSTERS WITH DBScan #####################################
    try:
        db = DBSCAN(eps=1.0 * max_dist_betweeen_gexagons
                    , min_samples=min_dbscan_samples
                    # , algorithm='ball_tree'
                    # , metric='haversine'
                    ).fit(X)
    except ValueError:
        db = DBSCAN(eps=1.6 * max_dist_betweeen_gexagons
                    , min_samples=min_dbscan_samples
                    # , algorithm='ball_tree'
                    # , metric='haversine'
                    ).fit(X)
        logger.error('DANGER!!!  EROROR IN DBSCAN PROCESS!!!')

    ### Добиваемся чтобы не было 0 кластера, кластеры начинаются с 1-го.
    labels_geoclusters = db.labels_ + 1
    ##################################################################

    df[labels] = labels_geoclusters

    return df

def make_clusters_from_geoclusters(df, features, labels, gex_in_diff_clust, min_gex_in_clust, logger):
    '''
    Разбиваем геокластер на несколько кластеров в соответствии с заданными
    признаками(features), и возвращаем датасет с переразмеченным столбцом
    кластеров

    df
    features  - список признаков испльзуемых KMeans
    labels  - название столбца в котором хранятся номера кластеров
    '''

    ### Формируем спиок геокластеров, с которыми будеи работать
    geoclusters_list = list(df['labels_geoclusters'].unique())

    for cluster in geoclusters_list:
        df_cluster = df[df['labels_geoclusters'] == cluster]
        ### Делим геокластер на кластеры если он состоит более чем из 10 гексагонов
        if df_cluster['hex_id'].nunique() >= gex_in_diff_clust:
            ### Filter and normilize data
            X = df_cluster[features]
            X_scale = StandardScaler().fit_transform(X)

            ### Silhouette number of clusters ######################################
            number_of_clusters_sil = find_num_clusters_silhouette(X_scale,logger)

            ### Корректируем количество кластеров
            number_of_clusters_sil_corr = min(number_of_clusters_sil,
                                              int(df_cluster['hex_id'].nunique() / min_gex_in_clust))

            ### Кластеризуем геокластеры на кластеры
            km = KMeans(n_clusters=number_of_clusters_sil_corr
                        # , init='k-means++'
                        # , max_iter=1000
                        , n_init=20
                        , random_state=17
                        ).fit(X_scale)
            labels_kmeans = km.predict(X_scale)

            ### Назначаем номера геокластеров с наследованием 21 - это 1-й кластер из 2 геокластера
            df.loc[df['labels_geoclusters'] == cluster, 'labels_geoclusters'] = \
                10 * df.loc[df['labels_geoclusters'] == cluster, 'labels_geoclusters'] + labels_kmeans

    return df

def calculate_geominimals(df, cluster, median_bill_multiplicator,logger):
    '''
    Функция рассчитывающая размер геоминималки в каждом кластере. Рассчитывается как медианная геоминималка * 1.2

    Input:
    df      - датасет
    cluster - номер кластера

    Output:
    df      - датасет
    '''

    df_clust = df[df['labels_geoclusters'] == cluster]

    df_non_null_clientbill = df_clust[df_clust['filtered_avg_driver_bill'] > 0]

    if (df_non_null_clientbill.shape[0] > 0):
        db_mean = df_non_null_clientbill['filtered_avg_driver_bill'].median()
    else:
        db_mean = 0

    db_mean = (5 * np.round((median_bill_multiplicator * db_mean) / 5)).astype(int)

    return db_mean

def calculate_median_driverbill(df, cluster,logger):
    '''
    Функция рассчитывающая размер медианного водительского чека в каждом кластере.
    Рассчитывается как медиана водительского чека.

    Input:
    df      - датасет
    cluster - номер кластера

    Output:
    df      - датасет
    '''

    df_clust = df[df['labels_geoclusters'] == cluster]

    df_non_null_clientbill = df_clust[df_clust['filtered_avg_driver_bill'] > 0]

    if (df_non_null_clientbill.shape[0] > 0):
        df_median_driverbill = df_non_null_clientbill['filtered_avg_driver_bill'].median()
    else:
        df_median_driverbill = 0

    return df_median_driverbill

def calculate_geomin_costs(df, median_bill_multiplicator, smooth_type, logger):
    '''
    Функция рассчитывающая геоминималки, медианный водит. чек, прогноз по количеству поездок в каждом кластере
    и формирующая geojson  со сглаженными координатами. Всё это сохраняется в датасет.


    Input:
      df          - датасет
    , smooth_type - тип сглаживания

    Output:
     cluster_info - датасет в разрезе кластер-город-день-час c информацией по геоминималкам
                    ,прогнозным поездкам, прогнозным затратам, сглаженным координатам.

    '''

    l_locality = []
    l_weekday = []
    l_hour = []

    l_rides_in_cluster = []
    l_geominimals = []
    l_median_driverbill = []
    l_predict_total_geomin_in_clust = []
    l_cnt_gexagons_in_cluster = []
    l_geojsons = []

    l_clusters = list(df.labels_geoclusters.unique())

    refinements = 2

    for cluster in l_clusters:

        locality = int(df[df['labels_geoclusters'] == cluster]['main_id_locality'].unique())
        weekday = int(df[df['labels_geoclusters'] == cluster]['weekday'].unique())
        hour = int(df[df['labels_geoclusters'] == cluster]['hour'].unique())

        #### Рассчет геоминималок в кластере
        geominimal = calculate_geominimals(df, cluster, median_bill_multiplicator,logger)

        #### Рассчет медианной цены водительского чека для этого кластера в данный день/час
        median_driverbill = calculate_median_driverbill(df, cluster,logger)

        predict_rides_in_clust = df[df.labels_geoclusters == cluster]['fact_rides'].sum()

        ### Если геоминималка >  среднего водительского чека, то учитываем разницу, иначе 0
        if (geominimal - median_driverbill > 0):
            predict_total_geomin_in_clust = predict_rides_in_clust * (geominimal - median_driverbill)
            # predict_total_geomin_in_clust = (5*np.round(predict_total_geomin_in_clust/5)).astype(int)
        else:
            predict_total_geomin_in_clust = 0

        cnt_gexagons_in_cluster = df[df['labels_geoclusters'] == cluster]['hex_id'].nunique()

        l_geominimals.append(geominimal)
        l_median_driverbill.append(median_driverbill)
        l_rides_in_cluster.append(predict_rides_in_clust)
        l_predict_total_geomin_in_clust.append(predict_total_geomin_in_clust)
        l_cnt_gexagons_in_cluster.append(cnt_gexagons_in_cluster)

        l_locality.append(locality)
        l_weekday.append(weekday)
        l_hour.append(hour)

        ### Convert H3 into border coordinates
        label_coord = []
        for hex_id in df['hex_id']:
            for coord in h3.h3_to_geo_boundary(hex_id):
                label_coord.append(coord)
        poly_points = pd.DataFrame(label_coord, columns=['lat', 'lon'])

        geodf = pd.DataFrame()
        geodf['geometry'] = [Polygon(h3.h3_to_geo_boundary(x, geo_json=True)) for x in
                             df[df.labels_geoclusters == cluster]["hex_id"]]
        gdf = gpd.GeoDataFrame(geodf['geometry'])

        if (smooth_type == 'no_smooth'):
            boundary = cascaded_union(gdf.geometry.values)
        elif (smooth_type == 'chaikin'):
            boundary = cascaded_union(gdf.geometry.values)
            try:
                boundary1 = np.array(boundary.exterior.coords)
                boundary2 = chaikins_corner_cutting(boundary1, refinements,logger) #refinements=2
                boundary = shapely_geometry.Polygon(boundary2)
            except AttributeError:
                boundary = cascaded_union(gdf.geometry.values)

        ##################

        geojson = str(GeoObject(boundary).geojson())

        l_geojsons.append(geojson)

        cluster_info = pd.DataFrame(
            zip(l_locality, l_weekday, l_hour, l_clusters, l_cnt_gexagons_in_cluster, l_geominimals,
                l_median_driverbill, l_rides_in_cluster, l_predict_total_geomin_in_clust, l_geojsons) \
            , columns=['locality', 'weekday', 'hour', 'cluster_name', 'cnt_gexagons', 'geominimal', 'median_driverbill',
                       'predict_rides', 'predict_total_geominimals', 'geojson'])

    return cluster_info

def find_gexagons_for_use(df, cnt_fact_rides, logger):
    '''
    Формирует список гексаготов локации, в которых было более cnt_fact_rides поездок, таким
    образом исключая гексагоны, в которых не было поездок - реки, леса, заводы и т.д.
    '''
    gexagons_for_use = list(df[df.fact_rides >= cnt_fact_rides]['hex_id'])

    return gexagons_for_use

def chaikins_corner_cutting(coords, refinements,logger):
    '''
    Функция для сглаживания углов по методу Чайкина с сохданениес площадей.

    Input:
    coords - список координат точек
    refinements - уровень сглаживания. Чем больше тем сильнее сглаживает.

    Output:
    coords - список координат точек после сглаживания
    '''
    coords = np.array(coords)
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords

def create_result_dataset(df,logger):
    '''
    Создаём результирующий датасет из промежуточного, отбирая нужные столбцы и добавляя недостающие.
    Этот датасет будет записываться в DWH для отображения в Таксисерве.
    '''

    result = pd.DataFrame()
    #result['locality'] = df['locality']
    result['weekday']         = df['weekday']
    result['hour']            = df['hour']
    result['labels']          = df['cluster_name']
    result['price']           = df['geominimal']

    result['predict_rides']   = df['predict_rides']
    result['predict_total_geominimals'] = df['predict_total_geominimals']
    result['cnt_gexagons']    = df['cnt_gexagons']
    result['polygon']         = df['geojson']
    #result["created_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #result["version"] = 'test_27_10_2021'
    return result
################################################################################3

def database_downloader(query, id_request, mysql, connection, logger):
    '''
    ### Загрузчик из кликхауса, пробует загрузить несколько раз прежде чем упадёт.

    query      - запрос с подставленными переменными
    id_request - id запроса для заполнения таблицы request
    mysql      - коннектор к базе для записи результатов
    connection - коннектор
    '''
    flag = 0
    flag2 = 0

    while ((flag != 1) & (flag2 <= 5)):
        try:
            df = connection.fetch(query)

            flag = 1
        except Exception as err:
            flag2 = flag2 + 1
            logger.error(f"Unexpected error in DWH downloading: {err}, {type(err)}")
            time.sleep(4)
            df = None
            pass

    if (flag != 1):
        logger.error(' !!!!!!!!!   DANGER! PROBLEM WITH DWH DOWNLOAD !!!!!!!! ')
        update_request_table(id_request, 'FAILED', mysql, logger)
        exit(1)

    return df

###############################################################################################################
def download_data(id_request, locality, calculation_date, cnt_days_for_download, query_d2v_from_surge, query_filtered_client_bill,
                  query_info_into_gexagon, clickhouse_main,  mysql, logger):
    '''
    Выгружаем данные по заданным скриптам за заданное количество дней назад начиная с заданной даты.
    Приводим типы данных в датасете для экономии памяти.

    Input:
      locality                    - id города
    , calculation_date            - дата рассчета
    , cnt_days_for_download       - количество дней в прошлое от даты рассчета, формирующих начальную дату выгрузки
    , query_info                  - SQL запрос в текстовом виде
    , query_filtered_client_bill  - SQL запрос в текстовом виде
    , query_info_into_gexagon     - SQL запрос в текстовом виде
    , connection                  - параметры подключения к DWH

    Output:
    data  - датасет ссо всеми данными

    '''
    ###################################################################################
    ### Загружаем данные из query_d2v_from_surge
    try:
        data = database_downloader(query_d2v_from_surge.format(calculation_date, cnt_days_for_download, locality)
                                   , id_request
                                   , mysql
                                   , clickhouse_main
                                   , logger)
    except Exception as err:
        logger.error(f"Unexpected error in download_data query_info.sql {err}, {type(err)}")
        update_request_table(id_request, 'FAILED', mysql,logger)
        exit(1)
    ### Проверяем наличие в датасете нужных столбцов и полноту датасета
    columns_list = ['hex_id', 'weekday', 'hour', 'main_id_locality']
    if((data.shape[0] > 0) & (data.shape[1] > 0) & (all(elem in data.columns for elem in columns_list) == True)):
        pass
    else:
        logger.error('Problem with dataset data in download_data', data.shape[0] , data.shape[1] )
        update_request_table(id_request, 'FAILED', mysql,logger)
        exit(1)

    data = data.replace('\\N', np.NaN)
    data.columns = data.columns.str.lower()
    ### Добавляем координаты (как центры гексагонов 8) и отфильтровываем гексагоны, в которыз
    ### будем раздавать геоминималки
    data = add_lat_lon_cols(data, 'hex_id',logger)

    data.weekday = data.weekday.astype('int16')
    data.hour = data.hour.astype('int16')
    data.main_id_locality = data.main_id_locality.astype('int32')
    data.uviews_agg     = data.uviews_agg.astype('float32')
    data.udrivers_agg   = data.udrivers_agg.astype('float32')
    data.d2v            = data.d2v.astype('float32')
    data.mean_surge_agg = data.mean_surge_agg.astype('float32')
    data.geo_lat        = data.geo_lat.astype('float32')
    data.geo_long       = data.geo_long.astype('float32')

    logger.info('Download query_info.sql rows:%10s cols:%10s size_Mb: %8s'
          , data.shape[0], data.shape[1], round(data.memory_usage().sum().sum() / 1024 ** 2, 3)
          )

    #############################################################
    try:
        df_clientbill = database_downloader(query_filtered_client_bill.format(calculation_date, cnt_days_for_download, locality)
                                          , id_request
                                          , mysql
                                          , clickhouse_main
                                          , logger)
    except Exception as err:
        logger.error(f"Unexpected error in query_filtered_client_bill.sql {err}, {type(err)}")
        update_request_table(id_request, 'FAILED', mysql,logger)
        exit(1)

    columns_list = ['weekday', 'hour', 'main_id_locality']
    if((df_clientbill.shape[0] > 0) & (df_clientbill.shape[1] > 0) & (all(elem in df_clientbill.columns for elem in columns_list) == True)):
        pass
    else:
        logger.error('Problem with dataset df_clientbill in download_data')
        update_request_table(id_request, 'FAILED', mysql,logger)
        exit(1)

    df_clientbill = df_clientbill.replace('\\N', np.NaN)
    df_clientbill.columns = df_clientbill.columns.str.lower()

    df_clientbill.weekday = df_clientbill.weekday.astype('int16')
    df_clientbill.hour = df_clientbill.hour.astype('int16')
    df_clientbill.main_id_locality = df_clientbill.main_id_locality.astype('int32')
    df_clientbill.filtered_avg_client_bill = df_clientbill.filtered_avg_client_bill.astype('float32')
    df_clientbill.filtered_avg_driver_bill = df_clientbill.filtered_avg_driver_bill.astype('float32')
    df_clientbill.filtered_avg_surge = df_clientbill.filtered_avg_surge.astype('float32')

    logger.info('Download query_filtered_client_bill.sql rows:%10s cols:%10s size_Mb: %8s'
          , df_clientbill.shape[0], df_clientbill.shape[1], round(df_clientbill.memory_usage().sum().sum() / 1024 ** 2, 3)
          )

    ##############################################################
    ### Выгружаем информацию о клиентах в каждом гексагоне
    try:
        df_gexagon = database_downloader(query_info_into_gexagon.format(calculation_date, cnt_days_for_download, locality)
                                          , id_request
                                          , mysql
                                          , clickhouse_main
                                          , logger)
    except Exception as err:
        logger.error(f"Unexpected error in query_info_into_gexagon.sql {err}, {type(err)}")
        update_request_table(id_request, 'FAILED', mysql,logger)
        exit(1)

    columns_list = ['weekday', 'hour', 'main_id_locality']
    if ((df_gexagon.shape[0] > 0) & (df_gexagon.shape[1] > 0) & (
            all(elem in df_gexagon.columns for elem in columns_list) == True)):
        pass
    else:
        logger.error('Problem with dataset df_gexagon in download_data')
        update_request_table(id_request, 'FAILED', mysql,logger)
        exit(1)

    df_gexagon = df_gexagon.replace('\\N', np.NaN)
    df_gexagon.columns = df_gexagon.columns.str.lower()

    df_gexagon.weekday = df_gexagon.weekday.astype('int16')
    df_gexagon.hour = df_gexagon.hour.astype('int16')
    df_gexagon.main_id_locality = df_gexagon.main_id_locality.astype('int32')
    df_gexagon.fact_orders = df_gexagon.fact_orders.astype('int32')
    df_gexagon.fact_rides = df_gexagon.fact_rides.astype('int32')

    logger.info('Download query_info_into_gexagon.sql rows:%10s cols:%10s size_Mb: %8s'
          , df_gexagon.shape[0], df_gexagon.shape[1], round(df_gexagon.memory_usage().sum().sum() / 1024 ** 2, 3)
          )

    ### Соединяем данные по просмотрам, поездкам и лимитам в единый датасет
    data = data.merge(df_gexagon[['weekday', 'hour', 'hex_id', 'fact_orders', 'fact_rides']]
                      , how='left'
                      , left_on=['weekday', 'hour', 'hex_id']
                      , right_on=['weekday', 'hour', 'hex_id']
                      )
    data = data.merge(df_clientbill
                      , how='left'
                      , left_on=['main_id_locality', 'weekday', 'hour', 'hex_id']
                      , right_on=['main_id_locality', 'weekday', 'hour', 'hex_id']
                      )

    data['fact_orders'] = data['fact_orders'].fillna(0)
    data['fact_rides'] = data['fact_rides'].fillna(0)

    data.fact_orders = data.fact_orders.astype('int32')
    data.fact_rides = data.fact_rides.astype('int32')

    data['filtered_avg_client_bill'] = data['filtered_avg_client_bill'].fillna(0)
    data['filtered_avg_driver_bill'] = data['filtered_avg_driver_bill'].fillna(0)
    data['filtered_avg_surge'] = data['filtered_avg_surge'].fillna(0)

    logger.info('Total dataset rows:%10s cols:%10s size_Mb: %8s'
          , data.shape[0], data.shape[1], round(data.memory_usage().sum().sum() / 1024 ** 2, 3)
          )
    return data


def incentives_evaluation(data, main_id_locality, median_bill_multiplicator, logger):
    '''
    Функция для каждого города, дня и часа находящая геокластера, далее кластеры, и вычисляющая размер геоминималок
    для этих кластеров.

    Input:
      data              - датасет
    , main_id_locality  - город
    , plot              - флаг отображения графиков

    Output:
    df_all_geominimals - датасет вида город-день-час-кластер-координаты-геоминималки
    '''
    df_all_geominimals = pd.DataFrame()
    cnt_fact_rides = 2

    ###Находим гексагоны, из которых в приципе были поездки за последнее время (т.е. исключаем реки и леса)
    gexagons_use_in_locality = find_gexagons_for_use(data[data.main_id_locality == main_id_locality]
                                                     , cnt_fact_rides,logger) #cnt_fact_rides=2

    data_grouped = data.groupby(['main_id_locality', 'weekday', 'hour'], as_index=False)
    for name, data_filtered in data_grouped:

        weekday = name[1]
        hour = name[2]
        locality = name[0]

        if (locality == main_id_locality):

            ### Выборка только для данной локации, дня и часа
            # data_local      = filter_dataframe_local(data, locality, weekday, hour)
            data_local = data[(data.main_id_locality == locality) & (data.weekday == weekday) & (data.hour == hour)]

            df = data_filtered.copy()

            ### Находим гексагоны которые остаются после выкидывания гексагонов попавших на реки
            ### озёра заводы и леса
            df = df[df['hex_id'].isin(gexagons_use_in_locality)]

            ##############################################################################
            ### НАХОДИМ ГЕОКЛАСТЕРЫ И ОБРАБАТЫВАЕМ ИХ

            if (df.shape[0] < 3):
                continue
            ### Находим геокластеры
            df = find_geoclusters(df, labels, min_dbscan_samples,logger)
            #### Удаляем гексагон, вокруг которого менее cnt_neighbors гексагонов из его кластера
            df = drop_single_hexagons(df, labels, logger, cnt_neighbors= 1) #cnt_neighbors=1

            if (df.shape[0] < 3):
                continue
            #### Заполняем дыры в геокластерах 2 раза
            df = fill_hole_in_cluster(df, data_local,logger, cnt_neighb_treshold=4)
            df = fill_hole_in_cluster(df, data_local,logger, cnt_neighb_treshold=4)

            ##############################################################################
            ### ДЕЛИМ ГЕОКЛАСТЕРЫ НА КЛАСТЕРЫ И ОБРАБАТЫВАЕМ
            features_geoclusters = ['geo_lat', 'geo_long', 'filtered_avg_driver_bill']

            ### Делим геокластер на несколько кластеров
            df = make_clusters_from_geoclusters(df, features_geoclusters, labels, 9, 7,logger)
            #### Переназначаем кластер гексагона на основе кластеров, которые его окружают.
            df = reassign_hexagon_by_neighbors(df, labels, logger, cnt_neighbors=3)
            #### Удаляем гексагон, вокруг которого менее cnt_neighbors гексагонов из его кластера
            df = drop_single_hexagons(df, labels,logger, cnt_neighbors=1)
            if (df.shape[0] < 3):
                continue

            #### Заполняем дыры в геокластерах
            df = fill_hole_in_cluster(df, data,logger, cnt_neighb_treshold=4)
            #### Удаляем гексагон, вокруг которого менее cnt_neighbors гексагонов из его кластера
            df = drop_single_hexagons(df, labels,logger, cnt_neighbors=1)

            ##############################################################################
            ### СОХРАНЯЕМ ДАННЫЕ В ЦИКЛЕ

            if (df.shape[0] < 3):
                continue

            cluster_info = calculate_geomin_costs(df, median_bill_multiplicator, 'chaikin',logger)

            df_all_geominimals = df_all_geominimals.append(cluster_info)

            #geominimals = str(list(cluster_info.sort_values('cnt_gexagons', ascending=False)['geominimal']))

            cnt_clusters = cluster_info['cluster_name'].nunique()
            cnt_gexagons = cluster_info['cnt_gexagons'].sum()
            sum_predict_rides = cluster_info['predict_rides'].sum()
            sum_predict_geominimals = int(cluster_info['predict_total_geominimals'].sum().round(2) )
            #sum_median_driverbill = cluster_info['median_driverbill'].mean()
            geominimal = int(cluster_info['geominimal'].mean() )

            logger.info('city %8s weekday %3s hour %3s cnt_clust %3s cnt_gex %3s pr_ride %4s pr_geomin %6s geomin %6s'
                  ,locality, weekday, hour, cnt_clusters, cnt_gexagons, sum_predict_rides, sum_predict_geominimals,geominimal
                  )

    return df_all_geominimals


########################################################################################################
### Для локации ищем D2V который соответствует бюджету
def find_coeff_for_filter_iteration(data, locality, city_incentives_limit
                                    , filter1_start
                                    , filter2
                                    , min_filter1
                                    , max_filter1
                                    , median_bill_multiplicator
                                    , logger
                                    ):
    '''
    Функция для поиска значения filter1 и filter2 точно удовлетворяющего нашшим ограничениям по лимиту средств
    расходуемых на город. Функция оценивает расходы на геоминималки при текущем значении fulter1, далее
    итерациями меняет значение filter1 контролируя соответствующие ему расходы, до тех пор пока не попадёт в
    заданное значение с 5% отклонением или не поймёт, что данного значения расходов достичь невозможно.

    Input:
      data                       - датасет
    , locality                   - город
    , city_incentives_limit      - сумма которую не должны превышать расходы
    , filter1_start              - начальное значение filter1
    , filter2                    - начальное значение filter2
    , min_filter1                - минимальное значение filter1
    , max_filter1                - максимальное значение filter1

    Output:

    filter1 - новое найденное значение filter1
    '''

    summary_spending = 0
    a0 = min_filter1
    b0 = max_filter1
    cycle_count = 0
    filter1 = filter1_start
    step = 0.05

    ### Оставляем гексагоны которые соответствуют фильтрам filter1 и filter2. Для этого
    ### группируем данные по локации и лню недели, находим значения для квантилей заданных фильтрами
    ### и отфильтровываем ненужные гексагоны.
    logger.info(f'Filter in dinf_coeff filter1 = %s and filter2=%s', filter1, filter2)
    data_gexagons_for_boost = get_filtered_data_daily_tr(data
                                                         , ['main_id_locality', 'weekday']
                                                         , quantile_views=filter2
                                                         , quantile_d2v=filter1
                                                         )
    df = incentives_evaluation(data_gexagons_for_boost, locality, median_bill_multiplicator,logger)

    ### Если количество гексагонов для буста более 0
    if (df.shape[0] > 0):
        ### Преобразуем в флоат для борьбы с переполнением
        summary_spending = df['predict_total_geominimals'].sum().astype('float64')

        ### Пытаемся подобрать фильтр, пока разность между лимитом и прогнозной мотивацией > 5%. Т.е.
    ### подбираем мотивацию с точностью 5%. Останавливаемся когда подбор закончен или намотали 12 циклов
    ### для одного города.
    while (np.abs(summary_spending - city_incentives_limit) > 0.05 * city_incentives_limit):

        ### Отфильтровываем гексагоны которые хотим бустить геоминималками.
        ### Если затраты больше лимита по городу понижаем фильтр и снижвем затраты,
        ### иначе повышаем фильтр и повышаем затраты

        ### Понижаем фильтр => повышаем сумму мотиваций
        if (summary_spending > city_incentives_limit):
            logger.info('spending > limit, filter1:%4s spending:%8s step:%6s'
                        ,filter1, summary_spending,  step)
            a = a0
            b = filter1
            step = np.abs(filter1 - a) / 2
            filter1 = filter1 - step

        ### Повышаем фильтр => снижаем сумму мотиваций
        else:
            logger.info('spending <= limit, filter1:%4s spending:%8s step:%6s'
                        ,filter1, summary_spending,  step)
            a = filter1
            b = b0
            step = np.abs(filter1 - b) / 2
            filter1 = filter1 + step

        a0 = a
        b0 = b

        ### Если выполнили более 10 циклов для 1 локации - выход
        if ((cycle_count >= 10) | (step < 0.001)):
            break

        ### Оставляем гексагоны которые соответствуют фильтрам filter1 и filter2. Для этого
        ### группируем данные по локации и лню недели, находим значения для квантилей заданных фильтрами
        ### и отфильтровываем ненужные гексагоны.
        logger.info(f'Filter in dinf_coeff filter1 = {filter1} and filter2={filter2}')
        data_gexagons_for_boost = get_filtered_data_daily_tr(data
                                                             , ['main_id_locality', 'weekday']
                                                             , quantile_views=filter2
                                                             , quantile_d2v=filter1
                                                             )
        df = incentives_evaluation(data_gexagons_for_boost, locality, median_bill_multiplicator,logger)

        ### Если количество гексагонов для буста более 0
        if (df.shape[0] > 0):
            ### Преобразуем в флоат для борьбы с переполнением
            summary_spending = df['predict_total_geominimals'].sum().astype('float64')
        else:
            logger.info('Cant find filter for limit', filter1)
            filter1 = filter1_start
            break

        cycle_count = cycle_count + 1

    ### Финальная проверка, если значение фильтра находится в дрпустимых пределах, то ОК,
    ### иначе возвращаем значение фильтра по умолчанию
    if((filter1 >= min_filter1) & (filter1 <= max_filter1)):
        pass
    else:
        filter1 = filter1_start

    return round(filter1, 6)
############################################################################################################
def find_budget_treshold_for_locality(df, locality, city_limits_list, logger):
    '''
    В разрезе городов находим значения фильтров filter1 и filter2 такие чтобы попадать в
    заданный бюджет по городу.

    Input:
    df               - датасет
    locality         - город
    city_limits_list - список ограничений по городу

    Output:
    filter1_treshold - найденное значение filter1
    filter2_treshold - найденное значение filter2
    '''

    l_locality = []
    l_filter1 = []
    l_filter2 = []

    ### Для каждого города в датасете
    locality = int(locality)

    ### Локальная копия датасета для данного города
    df_local = df[df.main_id_locality == locality]  # .copy()

    ### Достаём лимит на мотивацию для этого города
    city_incentives_limit = city_limits_list[0]

    ### Достаём начальные значения фильтров для этого города
    filter1 = city_limits_list[2]
    filter2 = city_limits_list[4]
    min_filter1 = city_limits_list[1]
    max_filter1 = city_limits_list[3]
    median_bill_multiplicator = city_limits_list[5]

    ### Для каждого города находим размер фильтра1 чтобы попадать в ограничения по мотивации.
    ### Возвращает treshold филдьтра как скаляр.
    filter1_treshold = find_coeff_for_filter_iteration(df_local
                                                       , locality
                                                       , city_incentives_limit
                                                       , filter1
                                                       , filter2
                                                       , min_filter1
                                                       , max_filter1
                                                       , median_bill_multiplicator
                                                       , logger
                                                       )
    filter2_treshold = filter2
    logger.info(f'Use filter1= {filter1_treshold} and filter1= {filter2_treshold} for locality=filter1= {locality}')

    return filter1_treshold, filter2_treshold
############################################################################################################
def update_request_table(id_request, message, mysql, logger):

    query_update_request_end = """
        UPDATE geominimal_suggest_request
        SET calculate_end_dt   = '{1}',
            status             = '{2}'
        WHERE id               = '{0}'
    """
    try:
        calculate_end_dt = datetime.now()
        mysql.execute(query_update_request_end.format(id_request, calculate_end_dt, message));
    except Exception as err:
        logger.error(f"Unexpected error! Cant write in  geominimal_suggest_request! {err}, {type(err)}")
        exit(1)
    return
############################################################################################################
############################################################################################################
def update_request_table_start(id_request, message, mysql, logger):

    query_update_request_end = """
        UPDATE geominimal_suggest_request
        SET calculate_start_dt   = '{1}',
            status             = '{2}'
        WHERE id               = '{0}'
    """
    try:
        calculate_end_dt = datetime.now()
        mysql.execute(query_update_request_end.format(id_request, calculate_end_dt, message));
    except Exception as err:
        logger.error(f"Unexpected error! Cant write in  geominimal_suggest_request! {err}, {type(err)}")
        exit(1)
    return
####################################################################################################

def data_quality_validation(df, calculation_date, cnt_days_for_download, locality, query, connection, logger):
    '''
    Функция проверяюшая качество данных и их наличие в базах данных за указанный период

    :param df:
    :param calculation_date:
    :param cnt_days_for_download:
    :param locality:
    :param query:
    :param connection:
    :return:
    '''
    try:
        df_data_source_no_problems_flag = connection.fetch(query.format(calculation_date, cnt_days_for_download, locality))
    except Exception as err:
        logger.error(f"Unexpected error in data_quality_validation {err}, {type(err)}")
        df_data_source_no_problems_flag = False


    ### Считаем что данные верны если:
    if ((df_data_source_no_problems_flag['flag_data_source_no_problems'][0] == 1) & \
            (df.shape[0] > 100) & \
           # (df.shape[1] == 17) & \
            (df['weekday'].nunique() == 7) & \
            (df['hour'].nunique() == 24) & \
            (df['hex_id'].nunique() >= 20) & \
            (df['geo_lat'].min() > 37.00) & \
            (df['geo_lat'].max() < 70.00) & \
            (df['geo_long'].min() > 15.00) & \
            (df['geo_long'].min() < 179.00) & \
            (df['fact_orders'].sum() > 100) & \
            (df['fact_rides'].sum() > 100) & \
            (df['d2v'].mean() > 0) & \
            (df['d2v'].mean() < 1000)
    ):
        result = 'ok'
    else:
        result = 'error'
        logger.error('Problem with data quality!!!')
        logger.error('df_data_source_no_problems_flag: %s', df_data_source_no_problems_flag['flag_data_source_no_problems'][0])
        logger.error('unique_weekdays     %s', df['weekday'].nunique())
        logger.error('unique_hours        %s', df['hour'].nunique())
        logger.error('unique_gexagons     %s', df['hex_id'].nunique())
        logger.error('min_geo_lat         %s', df['geo_lat'].min())
        logger.error('max_geo_lat         %s', df['geo_lat'].max())
        logger.error('min_geo_long        %s', df['geo_long'].min())
        logger.error('max_geo_long        %s', df['geo_long'].min())
        logger.error('sum_fact_orders     %s', df['fact_orders'].sum())
        logger.error('sum_fact_rides      %s', df['fact_rides'].sum())
        logger.error('min_d2v             %s', df['d2v'].min())
        logger.error('mean_d2v            %s', df['d2v'].mean())
        logger.error('max_d2v             %s', df['d2v'].max())

    return result

#################################################################################
def check_dataframe_health(df, id_request, columns_list, message, mysql, logger):
    #columns_list = ['locality_id', 'limit_sum', 'created_dt', 'train_period', 'id', 'status']
    if ((df.shape[0] > 0) & (df.shape[1] > 0) & (all(elem in df.columns for elem in columns_list) == True)):
        pass
    else:
        logger.error(message)
        update_request_table(id_request, 'FAILED', mysql,logger)
        exit(1)
    return

#################################################################################
def get_request_from_taxiserv(mysql, logger):
    '''
    Функция для выгрузки данных запроса от Taxiserv из таблицы в DWH

    :param mysql: - подключение к mysql
    :return: датасет из 1 строчки с данными запроса
    '''
    ### Загружаем данные из таблицы request в которой хранятся все запросы на рассчет
    query_mysql_geominimal_request = '''
    select id, locality_id, limit_sum, created_dt, created_by, train_period, status
    from geominimal_suggest_request
    where status = 'PENDING'
    order by created_dt desc
    limit 1
    '''
    try:
        ### Получаем данные из MySQL таблицы geominimal_suggest_request и конвертируем их
        df = mysql.fetch(query_mysql_geominimal_request)
    except Exception as err:
        logger.error(f"Unexpected error in download request data {err}, {type(err)}")
        exit(1)
    ### Если есть запросы на рассчет продолжаем, иначе прекращаем.
    if (df.shape[0] == 0):
        logger.debug(f'No pending rows!   {datetime.now()}')
        print('No pending rows!')
        exit()
    #check_dataframe_health(df, ['locality_id', 'limit_sum', 'created_dt', 'train_period', 'id', 'status'], 'Problem with dataset df_request')

    ### Проверяем наличие данных в выгруженном датасете
    columns_list = ['locality_id', 'limit_sum', 'created_dt', 'train_period', 'id', 'status']
    if ((df.shape[0] > 0) & (df.shape[1] > 0) & (
            all(elem in df.columns for elem in columns_list) == True)):
        pass
    else:
        logger.error('Problem with dataset df_request')
        exit(1)

    try:
        ### Достаём данные запуска и преобразовываем их к нужным типам
        locality = int(df['locality_id'])
        limit_sum = int(df['limit_sum'])
        request_datetime = df['created_dt'][0].to_pydatetime()
        taxiserv_username = int(df['created_by'][0])
        cnt_days_for_download = int(df['train_period'])
        id_request = int(df['id'])
        status = str(df['status'])
        calculation_date = str(request_datetime.date())
    except Exception as err:
        logger.error(f"Unexpected error in download request data {err}, {type(err)}")
        exit(1)
    logger.info(f'Servise get request: locality:{locality} limit:{limit_sum} user:{taxiserv_username} days:{cnt_days_for_download} id_request:{id_request}')

    ### Проверяем что параметры запроса находятся в разумных пределах
    flag = True
    if((limit_sum <= 0) | (limit_sum > 10000000)):
        flag = False
    if(taxiserv_username <= 0):
        flag = False
    if((cnt_days_for_download <= 0) | (cnt_days_for_download >= 100)):
        flag = False
    if((limit_sum <= 0) | (limit_sum > 10000000)):
        flag = False
    if((request_datetime <= pd.to_datetime('2021-12-01 00:00:00')) | (request_datetime >= pd.to_datetime('2031-12-01 00:00:00'))):
        flag = False

    if(flag == True):
        pass
    else:
        logger.error('Problem with request limits in get_request_from_taxiserv')
        exit(1)

    return df

#######################################################################
def print_setup_results(df, locality, logger):
    '''
    Выводим итоговые результаты сетапа
    '''
    cnt_clusters = df['labels'].nunique()
    cnt_gexagons = df['cnt_gexagons'].sum()
    sum_predict_rides = df['predict_rides'].sum()
    sum_predict_geominimals = df['predict_total_geominimals'].sum().round(2)
    request_id = df['request_id'].min()

    min_price = df['price'].min()
    median_price = df['price'].median()
    max_price = df['price'].max()

    logger.info(  'Final result city %7s request_id %7s cnt_gex %3s predict_rides %4s total_spend %6s min_geomin %4s med_geomin %4s max_geomin %4s'
          , locality, request_id, cnt_gexagons, sum_predict_rides, sum_predict_geominimals, min_price, median_price, max_price
          )
    return
