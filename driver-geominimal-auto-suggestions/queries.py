### Основной запрос который из таблицы от команды суржа достаёт данные по d2v и views в разрезах
### дня недели, часа, гексагона, города
query_d2v_from_surge = '''
---- Правильный запрос
select
  toDayOfWeek(toDateTime(agg.datetime_val, 'Europe/Moscow')) as weekday
, toHour(toDateTime(agg.datetime_val, 'Europe/Moscow'))      as hour
, locality_id                                            as main_id_locality
, h3ToString(h3_index)                                   as hex_id

, sum(ifNull(agg.uViews_agg,0) )                             as uviews_agg
, sum(ifNull(agg.uOpen_drivers_agg,0) )                      as udrivers_agg
, avg(ifNull(agg.d2v,0) )                                    as d2v
, avg(ifNull(agg.mean_surge,0) )/10                          as mean_surge_agg

from sandbox.d2v_geo_agg_v2 agg
where agg.locality_id = {2}
and (toDateTime(agg.datetime_val, 'Europe/Moscow') )  between (toDateTime('{0}', 'Europe/Moscow') - INTERVAL {1} DAY )
                                                          and (toDateTime('{0}', 'Europe/Moscow') - INTERVAL 1 DAY   )
and agg.period_mins = 60
and agg.d2v  <= 50
and agg.d2v   >  0
group by weekday, hour, hex_id, locality_id

--FORMAT TSVWithNames
'''

### Вытаскиваем данные в разрезе каждого гексагона/часа/дня по клиентским чекам.
### Самые дорогие и длинные поездки отфильтрованны, для более адекватного вычисления цены.
query_filtered_client_bill = '''
select
a.weekday as weekday
,a.hour as hour
,h3ToString(a.h3_8)           as hex_id
,min(main_id_locality)        as main_id_locality
,avg(a.client_bill)        as filtered_avg_client_bill
,avg(a.driver_bill)        as filtered_avg_driver_bill
,avg(1.0*a.coefficient/10) as filtered_avg_surge
from (
         select toDayOfWeek(OrderedDate)                              as weekday
              , toHour(OrderedDate)                                   as hour
              , geoToH3(toFloat64OrZero(longitude), toFloat64OrZero(latitude), 8) as h3_8
              , main_id_locality
              , id
              , id_client
              , status
              , exp_distance
              , exp_triptime
              , client_bill
              , driver_bill
              , coefficient
         from etl_city_import.order_closed_dist ocd
         inner join rt_city_import.tariff1 t1 on toInt64(ocd.tariff) = toInt64(t1.id)
         where OrderedDate between toDate('{0}') - {1} and toDate('{0}') - INTERVAL 1 DAY
           and isNotNull(longitude)
           and isNotNull(latitude)
           and t1.id_tariff_group = 2
           and main_id_locality = {2}
         ) a
where a.exp_distance < 25 --- distance less 25 km
  and a.coefficient  < 30 --- surge < 2.5 this is not mistake!
group by a.weekday ,a.hour ,a.h3_8
--FORMAT TSVWithNames
'''
### Информация о количестве фактических поездок в данных гексагонах в день недели и час дня
query_info_into_gexagon = '''
select
weekday
,hour
,h3ToString(h3_8) as hex_id
,min(main_id_locality) as main_id_locality
,count(distinct id)/1  as fact_orders
,count(distinct case when status = 'CP' then id else null end)/1 as fact_rides
from (
      select toDayOfWeek(OrderedDate)                              as weekday
           , toHour(OrderedDate)                                   as hour
           , geoToH3(toFloat64OrZero(longitude), toFloat64OrZero(latitude), 8) as h3_8
           , main_id_locality
           , id
           , id_client
           , status
           , exp_distance
           , exp_triptime
           , client_bill
           , coefficient
      from etl_city_import.order_closed_dist ocd
      inner join rt_city_import.tariff1 t1 on toInt64(ocd.tariff) = toInt64(t1.id)
      where
            (OrderedDate between toDate('{0}')-{1} and toDate('{0}') - INTERVAL 1 DAY  )
        and isNotNull(longitude)
        and isNotNull(latitude)
        and main_id_locality = {2}
        --and t1.id_tariff_group = 2
         ) a
where a.exp_distance < 25 --- distance less 25 km
  and a.coefficient  < 30 --- surge < 2.5 this is not mistake!         
group by weekday ,hour ,h3_8
--FORMAT TSVWithNames
'''

### Проверяем наличие данных за указанный период в 2-х базах данных.
### Положительный ответ будет только в случае наличия данных во всех базах.
query_check_problems_with_data_source = '''
select
case when sum(start_date_exist) = 2
      and sum(end_date_exist)   = 2
     then 1
     else 0
     end as flag_data_source_no_problems
from (
   --select
   -- (toDateTime('{0}', 'Europe/Moscow') - INTERVAL {1} DAY ) > min (toDateTime(datetime_val, 'Europe/Moscow') ) as start_date_exist
   --,(toDateTime('{0}', 'Europe/Moscow') - INTERVAL 1   DAY ) < max (toDateTime(datetime_val, 'Europe/Moscow') ) as end_date_exist
   --from sandbox.d2v_geo_agg_v2
   --where locality_id = {2}
   --  and period_mins = 60   
   --union all
    select
    min (calculated_dttm) <= toDate('{0}') - {1} as start_date_exist
        , max (calculated_dttm) >= toDate('{0}') - INTERVAL 1 DAY as end_date_exist
    from kafka_city_import.order_price_calculations
    where id_locality = {2}
    union all
    select
    min (OrderedDate) <= toDate('{0}') - {1} as start_date_exist
        , max (OrderedDate) >= toDate('{0}') - INTERVAL 1 DAY as end_date_exist
    from etl_city_import.order_closed_dist
    where main_id_locality = {2}
    )
--FORMAT TSVWithNames
'''
