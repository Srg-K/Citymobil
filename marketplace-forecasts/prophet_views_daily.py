import logging
import pandas as pd
import numpy as np
import db
import holidays
import warnings

from datetime import datetime
from fbprophet import Prophet
from sql_load_data_daily import que_views, que_rides, que_eta_rta, que_drivers_sh, que_oc_incent, que_weather, que_weather_forecast
from config import city_config

warnings.simplefilter('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def load_regressors(df, query, city, regressors):
    data = pd.DataFrame(exasol.execute(query.format(str(city))))
    data.columns = [x.lower() for x in data.columns]
    data = data.filter(['ds']+regressors)
    data['ds'] = pd.to_datetime(data['ds'])
    df = df.merge(data, how='left')
    return df


dt = datetime.now()
dt0 = dt.strftime("%Y-%m-%d %H")

logger = logging.getLogger()
mysql = db.get_mysql_conn()
exasol = db.get_exasol_connection()
clickhouse = db.get_ch_seg_conn()

holidays_dict = holidays.RU(years=(2019, 2020, 2021))
df_holidays = pd.DataFrame.from_dict(holidays_dict, orient='index').reset_index()
df_holidays = df_holidays.rename({'index': 'ds', 0: 'holiday'}, axis='columns')
df_holidays['ds'] = pd.to_datetime(df_holidays.ds)
df_holidays = df_holidays.sort_values(by=['ds'])
df_holidays = df_holidays.reset_index(drop=True)

for city in ['Санкт-Петербург','Москва', 'Воронеж']:
    
    df = pd.DataFrame(exasol.execute(que_views.format(str(city))))
    df.columns = ['ds', 'y']
    df['ds']= pd.to_datetime(df['ds'])
    df = df.drop(df.index[-1:])
    
    regressors = ['flag_pandemic', 'flag_anomaly', 'flag_holiday','precipitation_volume_modif', \
              'eta2rta', 'efficiency', 'avg_driver_bill','mean_surge_naive',  'avg_cancel_dist']
    
    for query in [que_rides, que_eta_rta, que_drivers_sh, que_oc_incent]:
        df = load_regressors(df, query, city, regressors)
    df = pd.concat([df, pd.DataFrame({'ds': pd.date_range(df['ds'].max(), periods=8, freq='D')})[1:]]).sort_values(by='ds')
    
    weather_hist = pd.DataFrame(clickhouse.fetch(que_weather.format(city_config[str(city)+'_locality'])))
    weather_forc = pd.DataFrame(clickhouse.fetch(que_weather_forecast.format(city_config[str(city)+'_locality'])))
    weather_hist['ds'] = weather_hist.ds.apply(lambda x: x.date())
    weather_forc['ds'] = weather_forc.ds.apply(lambda x: x.date())
    weather_hist = weather_hist[['ds', 'precipitation_volume']].groupby('ds').sum().reset_index()
    weather = pd.concat([weather_hist[:-1], weather_forc])
    weather.loc[weather.precipitation_volume<1, 'precipitation_volume_modif'] = 0
    weather.loc[(weather.precipitation_volume>=1)&(weather.precipitation_volume<20), 'precipitation_volume_modif'] = 1
    weather.loc[weather.precipitation_volume>=20, 'precipitation_volume_modif'] = 2
    weather['ds'] = pd.to_datetime(weather['ds'])
    df = df.merge(weather[['ds', 'precipitation_volume_modif']], how='left', on='ds')
    
    #special events flags
    df.loc[(df.ds>='2020-03-28')&(df.ds<'2020-05-29'), 'flag_pandemic'] = 1
    df.loc[(df.ds=='2020-02-24')|(df.ds=='2021-02-13')|(df.ds=='2021-02-14'), 'flag_anomaly'] = 1 
    df.loc[(df.ds>'2020-01-01')&(df.ds<'2020-01-13'), 'flag_holiday'] = 1
    df.loc[(df.ds>'2021-01-01')&(df.ds<'2021-01-13'), 'flag_holiday'] = 1
    df.loc[(df.ds>'2022-01-01')&(df.ds<'2022-01-09'), 'flag_holiday'] = 1
    df.loc[(df.ds>='2020-05-01')&(df.ds<'2020-05-10'), 'flag_holiday'] = 1
    df.loc[(df.ds>='2021-05-01')&(df.ds<'2021-05-10'), 'flag_holiday'] = 1
    df.loc[(df.ds>='2022-05-01')&(df.ds<'2022-05-09'), 'flag_holiday'] = 1

    for regressor in ['eta2rta', 'avg_driver_bill', 'mean_surge_naive']:
        df[str(regressor)+'_sh_7'] = df[regressor].shift(7)
        regressors.remove(regressor)
        regressors.append(str(regressor)+'_sh_7')
    for regressor in ['efficiency', 'avg_cancel_dist']:
        df[str(regressor)+'_sh_14'] = df[regressor].shift(14)
        regressors.remove(regressor)
        regressors.append(str(regressor)+'_sh_14')
    
    m = Prophet(
        yearly_seasonality=50,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=df_holidays,
        changepoint_prior_scale=0.45,
        seasonality_prior_scale=0.01,
        seasonality_mode='multiplicative',
        holidays_prior_scale = 0.05,
        interval_width=0.95
    ).add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=3
    ).add_seasonality(
        name="daily",
        period=1,
        fourier_order=20
    )
    for regressor in regressors:
        m.add_regressor(regressor)
    m.fit(df[:-7].fillna(0))
    future = m.make_future_dataframe(periods=7)
    future['ds'] = pd.to_datetime(future['ds'])
    future = future.merge(df.fillna(0), on = 'ds')
    future = future.fillna(0)
    fcst = m.predict(future)
    target_cols = ['yhat', 'yhat_lower', 'yhat_upper']
    forc = fcst.loc[:, target_cols]
    forc[forc < 0] = 0
    forc['ds'] = fcst['ds']

    forecast = forc[forc['ds'] >= dt0]
    forecast[target_cols] = forecast[target_cols].round(0).astype('int')
    forecast['created_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    forecast['version'] = "prophet_04"
    forecast['locality_id'] = city_config[str(city)+'_locality']
    result = forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'created_date', 'version','locality_id']]
    result.columns = ['date_day', 'views', 'views_lower', 'views_upper', 'created_date', 'version','locality_id']
    mysql.insert_data_frame("forecast_views_daily", result)
