select b.ID,
       a.SPLIT_NO,
       b.SEGMENT_TYPE,
       a.COUNT								                                as fact_rides,
       a.DRIVER_ID,
       a.PERIOD_DATE_START,
       a.PERIOD_ORDER_COUNT 		                                	    as target_x,
       a.PAID_SUM         			                                        as target_y,
       a.PAYMENT / 100                                                    	as budget,
       b.PERIOD
from REPLICA.driver_bonus_order_counts a
    left join
        (
            select ID, ID_LOCALITY, SEGMENT_TYPE, PERIOD
            from REPLICA.DRIVER_BONUS
            where DATE_START >= '2020-04-01 00:00:00'
        ) b
    on a.BONUS_ID = b.ID

    left join
        (
        select
        BONUS_ID
        from REPLICA.driver_bonus_order_counts
        group by BONUS_ID
        having count(distinct SPLIT_NO) < 4 and count(distinct DRIVER_ID) > 50
            ) d
    on a.BONUS_ID = d.BONUS_ID

where 1 = 1
    and b.ID is not null
    and d.BONUS_ID is not null
    and a.PERIOD_ORDER_COUNT > 0
    and b.ID_LOCALITY = {0}
