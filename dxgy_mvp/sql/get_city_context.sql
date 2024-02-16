select       to_date(oc.ORDEREDDATE)                                                                                                as "DATE"
             , count(oc.id)                                                                                                         as orders
             , count(distinct oc.DRIVER)                                                                                            as active_drivers
             , count(case when oc.EXP_DISTANCE > 6 then oc.id else null end) / count(*)                                             as share_long
             , avg(case when oc.EXP_DISTANCE > 6 then oc.EXP_PRICE end)                                                             as avg_price_long
             , avg(case when oc.EXP_DISTANCE <= 6 then oc.EXP_PRICE end)                                                            as avg_price_short
             , avg(case when oc.DRIVER_BILL > 0 and oc.EXP_DISTANCE > 6 then oc.DRIVER_BILL end)                                    as avg_price_driver_long
             , avg(case when oc.DRIVER_BILL > 0 and oc.EXP_DISTANCE <= 6 then oc.DRIVER_BILL end)                                   as avg_price_driver_short
             , sum(case when CLIENT_BILL_AMT > 0 then CLIENT_BILL_AMT else 1 end)                                                   as gmv
             , count(case when oc.STATUS = 'CP' then oc.id else null end)                                                           as rides
             , round(sum(COMMISSION_TOTAL_AMT) / sum(case when CLIENT_BILL_AMT > 0 then CLIENT_BILL_AMT else 1 end),4)              as commission
             , SUM(
                CI_CLIENT_REJECT + CI_COUPONS + CI_DISCOUNT_FOR_NEW_CLIENTS + CI_DISCOUNT_OTHER + CI_HR +
                CI_LOYALITY_PROGRAM_BONUS + CI_MAIL_COMBO + CI_MAIL_SPECIAL + CI_OLD_BONUS +
                --CI_PAID_BY_BONUS_POINTS +
                CI_PAID_PARTNER + CI_POTENTIAL + CI_PUSH_DISCOUNT + CI_SBER_PRIME + CI_SBER_SPASIBO_BONUS +
                DI_MOMENTAL) - SUM(di_welcome_dxgy + di_power_dxgy + di_other_dxgy + di_main_dxgy
                    + di_guaranteed_amt_per_hour + di_geo_minimal_amt + di_gold_minimal_amt
                    + di_base_minimal_amt + di_silver_minimal_amt + di_other_incentives_amt)                                        as OTHER_I
            , (sum(COMMISSION_TOTAL_AMT) - SUM(
                CI_CLIENT_REJECT + CI_COUPONS + CI_DISCOUNT_FOR_NEW_CLIENTS + CI_DISCOUNT_OTHER + CI_HR +
                CI_LOYALITY_PROGRAM_BONUS + CI_MAIL_COMBO + CI_MAIL_SPECIAL + CI_OLD_BONUS +
                --CI_PAID_BY_BONUS_POINTS +
                CI_PAID_PARTNER + CI_POTENTIAL + CI_PUSH_DISCOUNT + CI_SBER_PRIME + CI_SBER_SPASIBO_BONUS +
                DI_MOMENTAL) - SUM(di_mfg + di_welcome_dxgy + di_power_dxgy + di_other_dxgy + di_main_dxgy
                    + di_guaranteed_amt_per_hour + di_geo_minimal_amt + di_gold_minimal_amt
                    + di_base_minimal_amt + di_silver_minimal_amt + di_other_incentives_amt))                                       as contribution
        from REPLICA.ORDER_CLOSED oc
            LEFT JOIN REPLICA_MART.INCENTIVE_COMISSION ic on oc.id = ic.ORDER_ID
            where 1=1
            and oc.MAIN_ID_LOCALITY = {0}
            and to_date(oc.ORDEREDDATE) between '{1}' and '{2}'
        GROUP BY 1
