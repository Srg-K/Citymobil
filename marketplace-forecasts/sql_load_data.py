que_views = """
select
     date_trunc('hour',VIEW_DTTM) as dt, 
     SUM(VIEW_FIFTEEN_MIN_CNT)  as views
FROM REPLICA_MART.CALCULATION_AGGREGATED ca
WHERE VIEW_DTTM >= '2019-01-01'
    and ca.LOCALITY_RK = {0}
    and ca.TARIFF_RK in {1}
GROUP BY date_trunc('hour',VIEW_DTTM)
ORDER BY date_trunc('hour',VIEW_DTTM)
"""

que_drivers = """
SELECT
    date_trunc('hour',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0))) as dt,
    COUNT( distinct oc.Driver) as driver_count_order
FROM replica.order_closed oc
    inner join dds.s_company_account cmp1
        on oc.companyid = cmp1.company_account_rk
        and oc.ordereddate >=  cmp1.valid_from_dttm 
        and oc.ordereddate < cmp1.valid_to_dttm
    left join md.locality gl
        on oc.main_id_locality = gl.locality_rk
WHERE oc.ORDEREDDATE >= '2019-09-30'
    and oc.MAIN_ID_LOCALITY = {0}
    and oc.TARIFF in {1}
    AND oc.g_type = 0 /*Исключаем тестовые заказа*/
    AND oc.get_company <> 3 /*Тестовая компания*/
    AND cmp1.company_rk = 1 /*Платформа -Ситимобил*/
    AND oc.DRIVER > 0 /* Исключаем заказы с неназначенным водителем */
group by date_trunc('hour',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
order by date_trunc('hour',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
"""

que_rides = """
select
    date_trunc('hour',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0))) as dt, 
    SUM(case when oc.status='CP' then 1 else 0 end) as ride_cnt
from replica.order_closed oc
    inner join dds.s_company_account cmp1
        on oc.companyid = cmp1.company_account_rk
        and cmp1.valid_to_dttm = udf.inf()
    left join md.locality gl
        on oc.main_id_locality = gl.locality_rk
WHERE oc.ORDEREDDATE >= '2019-09-30'
    and oc.MAIN_ID_LOCALITY = {0}
    and oc.TARIFF in {1}
    and oc.COMPANYID in (1, 4570, 5746, 5810, 5918, 7754, 14430)
    AND oc.g_type = 0 /*исключаем тестовые заказаы*/
    AND oc.get_company <> 3 /*Тестовая компания*/
    AND cmp1.company_rk = 1 /*Платформа - Ситимобил*/
group by date_trunc('hour',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
order by date_trunc('hour',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
"""

que_weather = """
select
    toStartOfHour(time) as dt,
    temperature,
    temperature_human,
    precipitation_volume,
    cloudiness,
    humidity,
    weather_text
from segmentation.weather_history
where locality_id = {0}
order by time
"""

que_weather_forecast = """
select
    toStartOfHour(time) as dt,
    temperature,
    temperature_human,
    precipitation_volume,
    cloudiness,
    humidity,
    weather_text
from segmentation.weather_forecast_hourly a
inner join
(
    select max(created_date) max_date
    from segmentation.weather_forecast_hourly
    where locality_id = {0}
) b
on a.created_date = b.max_date
where a.locality_id = {0}
order by time
"""