import logging
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans
from datetime import datetime

import db
import config
import regions
from sql_data_kafka_rfm_new import que_1
from subsegm2segm import dict_3_segm, dict_5_segm

dt = datetime.now()
dt0 = dt.strftime("%Y-%m-%d")


def order_cluster(cluster_field_name, target_field_name, df, ascending):
    """This function order clusters in ascending way after kmeans algo """
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index": cluster_field_name})
    return df_final


def segmentation(X, segm_dict, n_clust=4):
    """This function calculate RFM segms,
    assigns total segment according to selected dict
    and adds its values to the input df as new cols """
    # temporary crutch to deal with new cities (less than full 30 days of data)
    if len(X) > 10:
        X_scaled = pd.DataFrame()
        for col in ['recency', 'frequency', 'monetary']:
            x_log = np.log(X[col].values + 1).reshape(-1, 1)
            scaler = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True).fit(x_log)
            X_scaled.loc[:, col] = pd.Series(scaler.transform(x_log).reshape(1, -1)[0], name=col)
            x = X_scaled[col].values.reshape(-1, 1)
            km = KMeans(n_clusters=n_clust, random_state=42)
            km.fit(x)
            X.loc[:, 'Cluster_' + col] = km.predict(x)
            if col == 'recency':
                X = order_cluster('Cluster_' + col, col, X, False)
            else:
                X = order_cluster('Cluster_' + col, col, X, True)
        X['rfm_str'] = X[['Cluster_recency', 'Cluster_frequency', 'Cluster_monetary']].apply(
            lambda x: ''.join(x.values.astype(str)), axis=1)
        X['Cluster_Total'] = X['rfm_str'].map(segm_dict)
        X.drop(columns=['rfm_str'], inplace=True)
        return X


def prepare_df(X):
    """prepare df by means of ordering cols and adjust their names"""
    res = X.loc[:, ['date', 'region_id', 'driver_id', 'Cluster_recency',
                    'Cluster_frequency', 'Cluster_monetary', 'Cluster_Total']]
    res.columns = ['date', 'locality_id', 'driver_id', 'recency_segment_id',
                   'frequency_segment_id', 'monetary_segment_id', 'segment_id']
    return res


logger = logging.getLogger()
mysql = db.get_mysql_conn()
clickhouse = db.get_ch_seg_conn()

regions_to_run_big_cities = regions.get_regions_to_run(2)
regions_to_run_small_cities = regions.get_regions_to_run(3)

regions_to_run = np.unique(regions_to_run_big_cities + regions_to_run_small_cities)
if len(regions_to_run) == 0:
    exit(0)

# start main part
data_full = clickhouse.fetch(que_1.format(dt0, tuple(regions_to_run)))
data_full = data_full.loc[:, ['rep_date', 'region_id', 'driver_id', 'recency', 'frequency', 'monetary']]
data_full.columns = ['date', 'region_id', 'driver_id', 'recency', 'frequency', 'monetary']
# fix data types
data_full['date'] = pd.to_datetime(data_full['date'])
int_cols = ['region_id', 'recency', 'frequency', 'monetary']
data_full[int_cols] = data_full[int_cols].apply(pd.to_numeric)

for i in regions_to_run_big_cities:
    type_id = 2
    X = data_full[data_full['region_id'] == i].copy()
    X = segmentation(X, dict_5_segm)
    mysql.insert_data_frame("driver_segmentation_rfm", prepare_df(X))
    mysql.insert_rows("driver_segmentation_dates",
                      [[type_id, i, dt0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), config.version]],
                      ["type_id", "locality_id", "date", "created_date", "version"]
                      )

for i in regions_to_run_small_cities:
    type_id = 3
    X = data_full[data_full['region_id'] == i].copy()
    X = segmentation(X, dict_3_segm)
    mysql.insert_data_frame("driver_segmentation_rfm_small_cities", prepare_df(X))
    mysql.insert_rows("driver_segmentation_dates",
                      [[type_id, i, dt0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), config.version]],
                      ["type_id", "locality_id", "date", "created_date", "version"]
                      )
