select to_date(o.ORDER_DTTM)                                                                                                dt,
       count(o.ORDER_RK)                                                                                                    trips,
       sum(case when ic.di_welcome_dxgy + ic.di_power_dxgy + ic.di_other_dxgy + ic.di_main_dxgy > 0
           and ic.di_mfg = 0
           and ic.DI_GEO_MINIMAL_AMT = 0 then 1 else 0 end)                                                                 dxgy_trips,
       sum(case when  ic.di_mfg > 0
           and ic.DI_GEO_MINIMAL_AMT = 0
           and ic.di_welcome_dxgy + ic.di_power_dxgy + ic.di_other_dxgy + ic.di_main_dxgy = 0 then 1 else 0 end)            mfg_trips,
       sum(case when  ic.DI_GEO_MINIMAL_AMT > 0
           and ic.di_welcome_dxgy + ic.di_power_dxgy + ic.di_other_dxgy + ic.di_main_dxgy = 0
           and ic.di_mfg = 0then 1 else 0 end)                                                                              geominimal_trips,
       sum(case when ic.di_welcome_dxgy + ic.di_power_dxgy + ic.di_other_dxgy + ic.di_main_dxgy > 0
                and  ic.di_mfg > 0
                and ic.DI_GEO_MINIMAL_AMT = 0 then 1 else 0 end)                                                           dxgy_mfg_trips,
       sum(case when ic.di_welcome_dxgy + ic.di_power_dxgy + ic.di_other_dxgy + ic.di_main_dxgy > 0
                and DI_GEO_MINIMAL_AMT > 0
                and ic.di_mfg = 0 then 1 else 0 end)                                                                        dxgy_geominimal_trips,
       sum(case when  ic.di_mfg > 0
                and DI_GEO_MINIMAL_AMT > 0
                and ic.di_welcome_dxgy + ic.di_power_dxgy + ic.di_other_dxgy + ic.di_main_dxgy = 0 then 1 else 0 end)       mfg_geominimal_trips,
       sum(case when ic.di_welcome_dxgy + ic.di_power_dxgy + ic.di_other_dxgy + ic.di_main_dxgy > 0
                and  ic.di_mfg > 0
                and DI_GEO_MINIMAL_AMT > 0 then 1 else 0 end)                                                               all_ic_trips,
       SUM(ic.di_welcome_dxgy + ic.di_power_dxgy + ic.di_other_dxgy + ic.di_main_dxgy)                                      dxgy,
       SUM(ic.di_mfg)                                                                                                       mfg,
       sum(DI_GEO_MINIMAL_AMT)                                                                                              geomoninmal,
       local.dxgy + local.mfg + local.geomoninmal                                                                           total
from EMART."ORDER" o
left join REPLICA_MART.INCENTIVE_COMISSION ic on ic.ORDER_ID = o.ORDER_RK
where o.STATUS_CD = 'CP'
    and to_date(o.ORDER_DTTM) between '2020-04-01' and '2021-09-30'
    and o.locality_rk = {0}
group by 1
