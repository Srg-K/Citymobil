que_views = """
select
  date_trunc('day', VIEW_DTTM) as ds
, sum(coalesce(VIEW_FIFTEEN_MIN_CNT, 0)) as views
from REPLICA_MART.CALCULATION_AGGREGATED as ca
     join (select distinct TARIFF_RK
        from EMART.TARIFF as t
        left join EMART.TARIFF_REPLACEMENT_RULE as trr
        on t.TARIFF_RK = trr.TARGET_TARIFF_RK
        where TARIFF_ZONE_NM = '{0}'
         and TARIFF_GROUP_NM in ('Эконом', 'Стандарт')
    ) trr  on ca.TARIFF_RK = trr.TARIFF_RK
where VIEW_DTTM >= '2020-01-01'
group by 1
order by 1
"""

que_drivers = """
SELECT
    date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0))) as ds,
    COUNT( distinct oc.Driver) as driver_count_order
FROM replica.order_closed oc
    inner join dds.s_company_account cmp1
        on oc.companyid = cmp1.company_account_rk
        and oc.ordereddate >=  cmp1.valid_from_dttm 
        and oc.ordereddate < cmp1.valid_to_dttm
    left join md.locality gl
        on oc.main_id_locality = gl.locality_rk
    join (select distinct TARIFF_RK
        from EMART.TARIFF as t
        left join EMART.TARIFF_REPLACEMENT_RULE as trr
        on t.TARIFF_RK = trr.TARGET_TARIFF_RK
        where TARIFF_ZONE_NM = '{0}'
         and TARIFF_GROUP_NM in ('Эконом', 'Стандарт')
    ) trr on oc.TARIFF = trr.TARIFF_RK
WHERE oc.ORDEREDDATE >= '2019-09-30'
    AND oc.g_type = 0 /*Исключаем тестовые заказа*/
    AND oc.get_company <> 3 /*Тестовая компания*/
    AND cmp1.company_rk = 1 /*Платформа -Ситимобил*/
    AND oc.DRIVER > 0 /* Исключаем заказы с неназначенным водителем */
group by date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
order by date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
"""


que_weather = """
select
    toStartOfHour(time) as ds,
    precipitation_volume   
from segmentation.weather_history
where locality_id = {0}
order by time
"""

que_weather_forecast = """
select
    time as ds,
    precipitation_volume
from segmentation.weather_forecast_daily a
inner join
(
    select max(created_date) max_date
    from segmentation.weather_forecast_daily
    where locality_id = {0}
) b
on a.created_date = b.max_date
where a.locality_id = {0}
order by time
"""

que_rides = '''
select
    date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0))) as ds
    ,avg(EXP_TRIPTIME) as mean_triptime_naive 
    ,avg(COEFFICIENT)  as mean_surge_naive 
    ,SUM(case when oc.CLIENTCHANCELATION is not null then 1 else 0 end)/count(oc.id) as perc_client_cancl 

   ,sum(oci.CANCEL_DISTANCE)/sum(case when oci.CANCEL_DISTANCE > 0 and oci.id>0 then 1 else null end) as avg_cancel_dist 
    
from replica.order_closed oc
    inner join dds.s_company_account cmp1
        on oc.companyid = cmp1.company_account_rk
        and cmp1.valid_to_dttm = udf.inf()
    left join md.locality gl on oc.main_id_locality = gl.locality_rk
    left join replica.order_closed_info oci on oc.id = oci.id and oc.DRIVER = oci.ID_DRIVER
    join (select distinct TARIFF_RK
        from EMART.TARIFF as t
        left join EMART.TARIFF_REPLACEMENT_RULE as trr
        on t.TARIFF_RK = trr.TARGET_TARIFF_RK
        where TARIFF_ZONE_NM = '{0}'
         and TARIFF_GROUP_NM in ('Эконом', 'Стандарт')
    ) trr on oc.TARIFF = trr.TARIFF_RK
WHERE oc.ORDEREDDATE >= '2019-09-30'
    and oc.COMPANYID in (1, 4570, 5746, 5810, 5918, 7754, 14430)
    AND oc.g_type = 0 /*исключаем тестовые заказаы*/
    AND oc.get_company <> 3 /*Тестовая компания*/
    AND cmp1.company_rk = 1 /*Платформа - Ситимобил*/
group by date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
order by date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
'''

que_drivers_sh = '''
select
date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0))) as ds

,SUM(sh.yellow_idle_next_order_sec + sh.home_idle_sec + sh.green_idle_sec
    + sh.on_trip_sec+ sh.to_client_collected_sec)/3600 as supply_hours 
,SUM(sh.on_trip_sec)/
  SUM(sh.yellow_idle_next_order_sec + sh.home_idle_sec + sh.green_idle_sec
    + sh.on_trip_sec+ sh.to_client_collected_sec) as efficiency 

from replica_mart.drivers_sh_orderly  sh
left join REPLICA.ORDER_CLOSED   oc    on sh.ORDER_RK = oc.ID and sh.DRIVER_RK = oc.DRIVER
inner join dds.s_company_account cmp1  on oc.companyid = cmp1.company_account_rk
left join md.locality gl  on oc.main_id_locality = gl.locality_rk
join (select distinct TARIFF_RK
        from EMART.TARIFF as t
        left join EMART.TARIFF_REPLACEMENT_RULE as trr
        on t.TARIFF_RK = trr.TARGET_TARIFF_RK
        where TARIFF_ZONE_NM = '{0}'
         and TARIFF_GROUP_NM in ('Эконом', 'Стандарт')
    ) trr on oc.TARIFF = trr.TARIFF_RK
WHERE oc.ORDEREDDATE >= '2019-09-30'
    and oc.ORDEREDDATE < current_date
    and oc.COMPANYID in (1, 4570, 5746, 5810, 5918, 7754, 14430)
    AND oc.g_type = 0 /*исключаем тестовые заказаы*/
    AND oc.get_company <> 3 /*Тестовая компания*/
    AND cmp1.company_rk = 1 /*Платформа - Ситимобил*/
group by date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
'''

que_oc_incent = '''
select
  date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))      as ds
    , count(case when oc.STATUS='CP' then oc.id else Null end)
        /SUM(case when oc.DRIVERASSIGNED is not Null  then 1 else null end) as Ad2R 
    , sum(case when oc.STATUS = 'CP' then oc.DRIVER_BILL else null end)
        / count(case when oc.STATUS = 'CP' then oc.id else Null end) as avg_driver_bill 

FROM REPLICA.ORDER_CLOSED   oc
left join replica_mart.incentive_comission      oi         on oc.id = oi.ORDER_ID
left join replica_mart.drivers_sh_orderly       sh         on oc.id = sh.order_rk
inner join dds.s_company_account cmp1  on oc.companyid = cmp1.company_account_rk
left join md.locality gl  on oc.main_id_locality = gl.locality_rk
join (select distinct TARIFF_RK
        from EMART.TARIFF as t
        left join EMART.TARIFF_REPLACEMENT_RULE as trr
        on t.TARIFF_RK = trr.TARGET_TARIFF_RK
        where TARIFF_ZONE_NM = '{0}'
         and TARIFF_GROUP_NM in ('Эконом', 'Стандарт')
    ) trr on oc.TARIFF = trr.TARIFF_RK

WHERE oc.ORDEREDDATE >= '2019-09-30'
    and oc.COMPANYID in (1, 4570, 5746, 5810, 5918, 7754, 14430)
    AND oc.g_type = 0 /*исключаем тестовые заказаы*/
    AND oc.get_company <> 3 /*Тестовая компания*/
    AND cmp1.company_rk = 1 /*Платформа - Ситимобил*/
GROUP BY  date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
'''

que_eta_rta = ''' 
select
date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0))) as ds
     , (case  when COUNT(case when oc.DriverAssigned is not Null then oc.id end) > 0
                   then SUM(oci.assign_dur) / COUNT(case when oc.DriverAssigned is not Null then oc.id end)
                 else 0
          end)/(SUM(case
          when    oc.Readyforcollection is not null and oc.Driverassigned is not null
                  and
                  minutes_between(oc.READYFORCOLLECTION, oc.DRIVERASSIGNED) < 60 /*исключаем выбросы, когда назначался водитель в другом конце города*/
              then minutes_between(oc.READYFORCOLLECTION, oc.DRIVERASSIGNED)
       end) / Count(case
                when    oc.Readyforcollection is not null and oc.Driverassigned is not null
                        and minutes_between(oc.READYFORCOLLECTION, oc.DRIVERASSIGNED) < 60
                    then oc.id end))    as eta2rta 

      from replica.order_closed oc
      left join dds.L_S_ORDER_STATUS_INFO oci on oci.ORDER_RK = oc.ID
      inner join dds.s_company_account cmp1  on oc.companyid = cmp1.company_account_rk
      left join md.locality gl  on oc.main_id_locality = gl.locality_rk
      join (select distinct TARIFF_RK
        from EMART.TARIFF as t
        left join EMART.TARIFF_REPLACEMENT_RULE as trr
        on t.TARIFF_RK = trr.TARGET_TARIFF_RK
        where TARIFF_ZONE_NM = '{0}'
         and TARIFF_GROUP_NM in ('Эконом', 'Стандарт')
    ) trr on oc.TARIFF = trr.TARIFF_RK
      WHERE oc.ORDEREDDATE >= '2019-09-30'
          and oc.COMPANYID in (1, 4570, 5746, 5810, 5918, 7754, 14430)
          AND oc.g_type = 0 /*исключаем тестовые заказаы*/
          AND oc.get_company <> 3 /*Тестовая компания*/
          AND cmp1.company_rk = 1 /*Платформа - Ситимобил*/
          AND oc.STATUS = 'CP'
     group by date_trunc('day',add_seconds(oc.ORDEREDDATE,coalesce(gl.time_offset,0)))
 ''' 