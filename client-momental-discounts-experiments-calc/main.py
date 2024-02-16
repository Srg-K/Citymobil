import query as q
import config
import utils
from stat_tests import StatTests
import pandas as pd
import time
import numpy as np
import db


def main(cfg=config, params=config.PARAMS):
    start_time = time.time()
    client_impacts = db.get_mysql_conn_db23().fetch(q.q_7())
    stat_tests = StatTests(cfg)
    split_col = cfg.SPLIT_COL

    for _, exp in client_impacts.iterrows():
        print(exp["ci_id"])
        df = utils.get_raw_data(
            ci_id=exp['ci_id'],
            dt_start=exp['date_started'],
            dt_end=exp['date_ended']
        )

        if df is None:
            continue

        exp_res = []
        totals_res = []
        for comb in utils.combine_splits(df, split_col):
            print(comb)
            df_with_single_combination = df[df[split_col].isin(comb)]
            totals = utils.get_totals(df=df_with_single_combination, ci_id=exp['ci_id'], split_col=split_col,
                                      split_pair=comb)
            totals_res.append(totals)

            for metric in params['metrics']:
                print(metric)
                df_compressed = utils.compress(
                    df=df_with_single_combination,
                    metric=metric,
                    split_col=params['metrics'][metric]['split_col'],
                    to_compress=params['metrics'][metric]['to_compress'],
                    filter_to=params['metrics'][metric]['filter_to'],
                    agg=params['metrics'][metric]['agg']
                )
                exp_res.append(
                    stat_tests.stat_inference(
                        df=df_compressed,
                        split_col=params['metrics'][metric]["split_col"],
                        metric_col=params['metrics'][metric]["metric_col"],
                        estimator=params['metrics'][metric]["estimator"],
                        metric=metric,
                        split_pair=comb,
                        post_power=params['metrics'][metric]["post_power"]
                    )
                )
        try:
            exp_res = stat_tests.multiple_comparison(pd.concat(exp_res))
            df_to_db = pd.concat(totals_res).merge(exp_res, on=['metric', 'split_pair'], how="left")
            df_to_db[['split_0', 'split_1']] = df_to_db[['split_0', 'split_1']].fillna(0)
        except Exception as e:
            print("no res to concat")
            continue
        utils.upload_to_db(df=df_to_db, ci_id=exp["ci_id"])
        print("time", time.time() - start_time)

main()
