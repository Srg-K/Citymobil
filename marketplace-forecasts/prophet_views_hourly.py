import logging
import pandas as pd
import numpy as np
import db
import holidays
import warnings

from datetime import datetime
from fbprophet import Prophet
from sql_load_data import que_views
from config import city_config

warnings.simplefilter('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


dt = datetime.now()
dt0 = dt.strftime("%Y-%m-%d %H")

logger = logging.getLogger()
mysql = db.get_mysql_conn()
exasol = db.get_exasol_connection()

holidays_dict = holidays.RU(years=(2019, 2020, 2021))
df_holidays = pd.DataFrame.from_dict(holidays_dict, orient='index').reset_index()
df_holidays = df_holidays.rename({'index': 'ds', 0: 'holiday'}, axis='columns')
df_holidays['ds'] = pd.to_datetime(df_holidays.ds)
df_holidays = df_holidays.sort_values(by=['ds'])
df_holidays = df_holidays.reset_index(drop=True)

for city in ['saint_p','moscow']:
    df = pd.DataFrame(exasol.execute(que_views.format(city_config[str(city)+'_locality'],city_config[str(city)+'_tarif'])))
    df.columns = ['ds', 'y']
    df['ds']= pd.to_datetime(df['ds'])
    df = df.drop(df.index[-1:])

    # train = df[df['ds'] < '2020-11-02']
    # test = df[(df['ds'] >= '2020-11-02') & (df['ds'] < '2020-11-09')]

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=df_holidays,
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=0.1,
        seasonality_mode='additive',
        interval_width=0.95
    ).fit(df)
    future = m.make_future_dataframe(periods=170, freq='H')
    fcst = m.predict(future)

    # mape = mean_absolute_percentage_error(test['y'].astype('float'), fcst[fcst['ds'] >= '2020-11-02']['yhat'])
    # print('MAPE: %.1f percent' % mape)

    target_cols = ['yhat', 'yhat_lower', 'yhat_upper']
    forc = fcst.loc[:, target_cols]
    forc[forc < 0] = 0
    forc['ds'] = fcst['ds']

    forecast = forc[forc['ds'] >= dt0]
    forecast[target_cols] = forecast[target_cols].round(0).astype('int')
    forecast['created_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    forecast['version'] = "prophet_01"
    forecast['locality_id'] = city_config[str(city)+'_locality']
    result = forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'created_date', 'version','locality_id']]
    result.columns = ['date_hour', 'views', 'views_lower', 'views_upper', 'created_date', 'version','locality_id']
    # print(result.head())
    mysql.insert_data_frame("forecast_views_hourly", result)
