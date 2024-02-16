select distinct d.ID 						AS driver_id
              , min(to_date(o.ORDER_DTTM)) 			as dt_first
              , max(to_date(o.ORDER_DTTM)) 			as dt_last
              , count(distinct ORDER_RK) 				as trips
from REPLICA.DRIVERS d
left join EMART."ORDER" o on d.ID = o.DRIVER_RK
where 1 = 1
and LOCALITY_RK = {0}
and to_date(o.ORDER_DTTM) < '{1}'
and STATUS_CD = 'CP'
group by 1