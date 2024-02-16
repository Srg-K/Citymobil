import logging
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import mannwhitneyu, norm, chi2_contingency
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.power import tt_ind_solve_power

import db
from query_1 import q_1


def get_bootstrap(
        data_column_1,  # числовые значения первой выборки
        data_column_2,  # числовые значения второй выборки
        boot_it=1000,  # количество бутстрэп-подвыборок
        statistic=np.mean,  # интересующая нас статистика
        bootstrap_conf_level=0.95  # уровень значимости
):
    boot_data = []
    for i in range(boot_it):
        samples_1 = data_column_1.sample(
            len(data_column_1),
            replace=True
        ).values

        samples_2 = data_column_2.sample(
            len(data_column_2),
            replace=True
        ).values

        boot_data.append(statistic(samples_1) - statistic(samples_2))

    pd_boot_data = pd.DataFrame(boot_data)

    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])

    p_1 = norm.cdf(
        x=0,
        loc=np.mean(boot_data),
        scale=np.std(boot_data)
    )
    p_2 = norm.cdf(
        x=0,
        loc=-np.mean(boot_data),
        scale=np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2

    return {"boot_data": boot_data,
            "quants": quants,
            "p_value": p_value}


def multi_func(targ):
    if '_' in targ:
        var_list = [int(i) for i in targ.split('_')]
        return max(var_list)
    else:
        return int(targ)


dt = datetime.now()
dt0 = dt.strftime("%Y-%m-%d")

logger = logging.getLogger()

mysql_prod = db.get_mysql_conn_prod()
mysql_segm = db.get_mysql_conn_segm()
exasol = db.get_exasol_connection()

TABLE_NAME = "driver_bonus_experiment_ab_tests"
BONUS_IDS = pd.DataFrame(exasol.execute('''
select distinct ID
from REPLICA.driver_bonus
where 1=1
and IS_EXPERIMENT = 1
and IS_ACTIVE = 1
'''))

for bonus_id in BONUS_IDS['ID'].tolist():
    df = pd.DataFrame(exasol.execute(q_1.format(bonus_id)))

    if len(df) < 10:
        continue

    splits = sorted(df['SPLIT_NO'].unique().tolist())
    result = pd.DataFrame(columns=['date', 'bonus_id', 'metric', 'split_1', 'split_2', 'p_value', 'p_adj', 'mde'])

    if len(splits) == 2:
        metric = 'tpd'
        tpd_0 = df[df['SPLIT_NO'] == 0]['COUNT']
        tpd_1 = df[df['SPLIT_NO'] == 1]['COUNT']
        try:
            tpd_p = mannwhitneyu(tpd_0, tpd_1).pvalue
        except ValueError:
            tpd_p = 0
            # logger.error("can't calculate metric {0} for bon_id {1}".format(metric, bonus_id))
        res = {'date': dt0, 'bonus_id': bonus_id, 'metric': metric,
               'split_1': 0, 'split_2': 1,
               'p_value': tpd_p, 'p_adj': np.NaN, 'mde': np.NaN}
        result = result.append(res, ignore_index=True)

        ratio = df[df['SPLIT_NO'] == 1]['DRIVER_ID'].nunique() / df[
            df['SPLIT_NO'] == 0]['DRIVER_ID'].nunique()

        result['mde'] = tt_ind_solve_power(alpha=0.01, power=0.8,
                                           nobs1=int(df[df['SPLIT_NO'] == 0]['DRIVER_ID'].nunique()),
                                           ratio=ratio)

    elif len(splits) == 3:
        metric = 'tpd'
        splits_comb = ((0, 1), (0, 2), (1, 2))
        for tup in splits_comb:
            tpd_0 = df[df['SPLIT_NO'] == tup[0]]['COUNT']
            tpd_1 = df[df['SPLIT_NO'] == tup[1]]['COUNT']
            try:
                tpd_p = mannwhitneyu(tpd_0, tpd_1).pvalue
            except ValueError:
                tpd_p = 0
                # logger.error("can't calculate metric {0} for bon_id {1}".format(metric, bonus_id))
            res = {'date': dt0, 'bonus_id': bonus_id, 'metric': metric,
                   'split_1': tup[0], 'split_2': tup[1],
                   'p_value': tpd_p, 'p_adj': np.NaN, 'mde': np.NaN}
            result = result.append(res, ignore_index=True)

        metric = 'cpt'
        for tup in splits_comb:
            cpt_0 = df[df['SPLIT_NO'] == tup[0]]['PAID_SUM']
            cpt_1 = df[df['SPLIT_NO'] == tup[1]]['PAID_SUM']
            try:
                cpt_p = get_bootstrap(cpt_0, cpt_1)['p_value']
            except TypeError:
                cpt_p = 0
                # logger.error("can't calculate metric {0} for bon_id {1}".format(metric, bonus_id))
            res = {'date': dt0, 'bonus_id': bonus_id, 'metric': metric,
                   'split_1': tup[0], 'split_2': tup[1],
                   'p_value': cpt_p, 'p_adj': np.NaN, 'mde': np.NaN}
            result = result.append(res, ignore_index=True)

        metric = 'AR'
        AR_21 = len(df[(df['SPLIT_NO'] == 1) & (df['COUNT'] > 0)])
        AR_11 = len(df[df['SPLIT_NO'] == 1])
        AR_22 = len(df[(df['SPLIT_NO'] == 2) & (df['COUNT'] > 0)])
        AR_12 = len(df[df['SPLIT_NO'] == 2])
        totals = np.array([AR_11, AR_12])
        successes = np.array([AR_21, AR_22])
        try:
            p_val = chi2_contingency(np.array([totals - successes, successes]))[1]
        except ValueError:
            p_val = 0
            # logger.error("can't calculate metric {0} for bon_id {1}".format(metric, bonus_id))
        res = {'date': dt0, 'bonus_id': bonus_id, 'metric': metric,
               'split_1': 1, 'split_2': 2, 'p_value': p_val, 'p_adj': np.NaN, 'mde': np.NaN}
        result = result.append(res, ignore_index=True)

        metric = 'SR'
        SR_21 = len(df[(df['SPLIT_NO'] == 1) & (df['COUNT'] >= df['X'].apply(multi_func))])
        SR_11 = len(df[df['SPLIT_NO'] == 1])
        SR_22 = len(df[(df['SPLIT_NO'] == 2) & (df['COUNT'] >= df['X'].apply(multi_func))])
        SR_12 = len(df[df['SPLIT_NO'] == 2])
        totals = np.array([SR_11, SR_12])
        successes = np.array([SR_21, SR_22])
        try:
            p_val = chi2_contingency(np.array([totals - successes, successes]))[1]
        except ValueError:
            p_val = 0
            # logger.error("can't calculate metric {0} for bon_id {1}".format(metric, bonus_id))
        res = {'date': dt0, 'bonus_id': bonus_id, 'metric': metric,
               'split_1': 1, 'split_2': 2, 'p_value': p_val, 'p_adj': np.NaN, 'mde': np.NaN}
        result = result.append(res, ignore_index=True)

        for tup in splits_comb:
            ratio = df[df['SPLIT_NO'] == tup[1]]['DRIVER_ID'].nunique() / df[
                df['SPLIT_NO'] == tup[0]]['DRIVER_ID'].nunique()

            result.loc[(result['split_1'] == tup[0]) & (result['split_2'] == tup[1]),
                       ['mde']] = tt_ind_solve_power(alpha=0.01, power=0.8,
                                                     nobs1=int(df[df['SPLIT_NO'] == tup[0]]['DRIVER_ID'].nunique()),
                                                     ratio=ratio)

    else:
        print('sorry, wrong number of splits')

    result['p_adj'] = pd.Series(dtype=float)
    result.loc[result['p_value'].notna(), ['p_adj']] = multipletests(result['p_value'][result['p_value'].isna() == False], alpha=0.05, method='fdr_bh')[1]

    mysql_segm.insert_data_frame(TABLE_NAME, result)

