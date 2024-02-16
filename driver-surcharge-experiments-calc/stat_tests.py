import pandas as pd
import numpy as np
import config
import random
import logging
from scipy.stats import norm, ttest_ind, mannwhitneyu, t
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


class StatTests(object):
    """Класс статистических методов"""

    def __init__(
            self,
            cfg: config
    ):
        """

        Parameters
        ----------
        cfg
            simulation_num - количество симуляций
            percent_sample - доля наблюдений для подвыборки из исходной выборки
            max_n_sample – ограничение по количеству наблюдений в подвыборке
            confidence_level – уровень значимости
            boot_it – количество бутстрап итераций
        """

        self.simulation_num: int = cfg.SIMULATION_NUM
        self.percent_sample: float = cfg.PERCENT_SAMPLE
        self.max_n_sample: int = cfg.MAX_N_SAMPLE
        self.confidence_level: float = cfg.CONFIDENCE_LEVEL
        self.boot_it: int = cfg.BOOT_ITERATIONS

    def fpr(self, df_x: pd.DataFrame, df_y: pd.DataFrame, metric_col: str, estimator: str, *args, **kwargs):

        """
        Расчет FPR для A/A-теста (или TPR для A/B)

        Parameters
        ----------
        df_x : pd.DataFrame
            Таблица данных эксперимента первого сплита X
        df_y : pd.DataFrame
            Таблица данных эксперимента второго сплита Y
        metric_col : str
            Наименование показателя для проверки
        estimator : {'t_test', 'prop', 'bootstrap', 'mann_whitney_test', 't_test_ratio'}
            Статистический критерий
        *args
        **kwargs

        Returns
        -------
        out : dict
            Результаты многократного расчета pvalue,
            средних и итоговый показатель FPR/TPR.
        """
        stat_res = {
            'aa': {'pvalue': []},
            'fpr': {f'fpr_{self.confidence_level}': .0}
        }

        estimators = {
            't_test': self.t_test,
            't_test_ratio': self.t_test_ratio,
            'prop': self.prop_test,
            'mann_whitney_test': self.mann_whitney_test,
            'bootstrap': self.bootstrap
        }

        for sim in range(self.simulation_num):
            if estimator == 't_test_ratio':
                metric_col = None
            x = self.make_sample(df_x, metric_col)
            y = self.make_sample(df_y, metric_col)
            pvalue, _, __, ___ = estimators[estimator](x, y, *args, **kwargs)
            stat_res['aa']['pvalue'].append(pvalue)
        stat_res['fpr'][f'fpr_{self.confidence_level}'] = self.compute_fpr(stat_res['aa']['pvalue'])

        return stat_res

    def t_test_ci(self, x, y, confidence_level=config.CONFIDENCE_LEVEL, *args, **kwargs):

        n1 = x.size
        n2 = y.size
        m1 = np.mean(x)
        m2 = np.mean(y)

        v1 = np.var(x, ddof=1)
        v2 = np.var(y, ddof=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            pooled_se = np.sqrt(v1 / n1 + v2 / n2)
            delta = m1 - m2
            tstat = delta / pooled_se
            df = ((v1 / n1 + v2 / n2) ** 2 / (v1 ** 2 / (n1 ** 2 * (n1 - 1)) + v2 ** 2 / (n2 ** 2 * (n2 - 1))))

        p = 2 * t.cdf(-abs(tstat), df)

        lb = delta - t.ppf((1 + confidence_level) / 2., df) * pooled_se
        ub = delta + t.ppf((1 + confidence_level) / 2., df) * pooled_se

        return tstat, p, delta, lb, ub

    def _ttest_ind(self, x, y, *args, **kwargs):
        return ttest_ind(x, y, *args, **kwargs), None, None, None

    def t_test(self, x, y, test="t_test_ci", *args, **kwargs):
        tests = {
            'ttest_ind': self._ttest_ind,
            't_test_ci': self.t_test_ci
        }
        try:
            _, pvalue, delta, lb, ub = tests[test](x[~np.isnan(x)], y[~np.isnan(y)], *args, **kwargs)
        except ValueError as e:
            logging.error(e)
            return None
        return pvalue, delta, lb, ub

    def mann_whitney_test(self, x, y, *args, **kwargs):
        try:
            _, pvalue = mannwhitneyu(x[~np.isnan(x)], y[~np.isnan(y)], *args, **kwargs)
        except ValueError as e:
            logging.error(e)
            return None
        return pvalue, None, None, None

    def t_test_ratio(self, df_0, df_1, confidence_level=config.CONFIDENCE_LEVEL):
        """t-Критерий с использованием delthamethod преобразования"""
        x_0 = np.array(df_0['x'])
        x_1 = np.array(df_1['x'])
        y_0 = np.array(df_0['y'])
        y_1 = np.array(df_1['y'])
        n_0 = np.sum(np.array(df_0['n']))
        n_1 = np.sum(np.array(df_1['n']))

        mean_x_0 = x_0.sum() / n_0
        mean_x_1 = x_1.sum() / n_1
        mean_y_0 = y_0.sum() / n_0
        mean_y_1 = y_1.sum() / n_1
        var_x_0 = np.sum(np.array([abs(a - mean_x_0) ** 2 for a in x_0])) / n_0
        var_x_1 = np.sum(np.array([abs(a - mean_x_1) ** 2 for a in x_1])) / n_1
        var_y_0 = np.sum(np.array([abs(a - mean_y_0) ** 2 for a in y_0])) / n_0
        var_y_1 = np.sum(np.array([abs(a - mean_y_1) ** 2 for a in y_1])) / n_1
        cov_0 = np.sum((x_0 - mean_x_0.reshape(-1, 1)) * (y_0 - mean_y_0.reshape(-1, 1)), axis=1) / n_0
        cov_1 = np.sum((x_1 - mean_x_1.reshape(-1, 1)) * (y_1 - mean_y_1.reshape(-1, 1)), axis=1) / n_1

        var_0 = var_x_0 / mean_y_0 ** 2 + var_y_0 * mean_x_0 ** 2 / mean_y_0 ** 4 - 2 * mean_x_0 / mean_y_0 ** 3 * cov_0
        var_1 = var_x_1 / mean_y_1 ** 2 + var_y_1 * mean_x_1 ** 2 / mean_y_1 ** 4 - 2 * mean_x_1 / mean_y_1 ** 3 * cov_1

        with np.errstate(divide='ignore', invalid='ignore'):
            df = ((var_0 / n_0 + var_1 / n_1) ** 2 / (
                        var_0 ** 2 / (n_0 ** 2 * (n_0 - 1)) + var_1 ** 2 / (n_1 ** 2 * (n_1 - 1))))[0]
            rto_0 = np.sum(x_0) / np.sum(y_0)
            rto_1 = np.sum(x_1) / np.sum(y_1)
            delta = rto_1 - rto_0
            pooled_se = np.sqrt(var_0 / n_0 + var_1 / n_1)[0]
            statistic = delta / pooled_se

        pvalue = 2 * t.cdf(-abs(statistic), df)

        lb = delta - t.ppf((1 + confidence_level) / 2., df) * pooled_se
        ub = delta + t.ppf((1 + confidence_level) / 2., df) * pooled_se

        return pvalue, delta, lb, ub

    def prop_test_ci(successes, counts, confidence_level=config.CONFIDENCE_LEVEL, *args, **kwargs):
        success_a, success_b = successes
        size_a, size_b = counts

        prop_a = success_a / size_a
        prop_b = success_b / size_b
        var_a = prop_a * (1 - prop_a) / size_a
        var_b = prop_b * (1 - prop_b) / size_b
        se = np.sqrt(var_a + var_b)

        delta = prop_b - prop_a
        z = norm(loc=0, scale=1).ppf((1 + confidence_level) / 2.)
        statistic = delta / se
        pvalue = 2 * np.minimum(norm(0, 1).cdf(statistic), 1 - norm(0, 1).cdf(statistic))

        lb = delta + -1 * z * se
        ub = delta + 1 * z * se
        return statistic, pvalue, delta, lb, ub

    def _proportions_ztest(self, counts, nobs, *args, **kwargs):
        return proportions_ztest(counts, nobs, *args, **kwargs), None, None, None

    def prop_test(self, x, y, test="prop_test_ci", *args, **kwargs):
        tests = {
            "proportions_ztest": self._proportions_ztest,
            "prop_test_ci": self.prop_test_ci,
        }
        counts = np.array([sum(x), sum(y)])
        nobs = np.array([len(x), len(y)])
        try:
            _, pvalue = tests[test](counts, nobs, *args, **kwargs)
        except ValueError as e:
            logging.error(e)
            return None
        return pvalue, None, None, None

    def bootstrap(self, x, y):
        boot_dist = []
        for sim in range(self.simulation_num):
            xb = np.mean(random.choice(x))
            yb = np.mean(random.choice(y))
            boot_dist.append(yb - xb)
        try:
            pvalue = min(
                norm.cdf(0, np.mean(boot_dist), np.std(boot_dist)),
                norm.cdf(0, -np.mean(boot_dist), np.std(boot_dist))
            ) * 2
        except ValueError as e:
            logging.error(e)
            return None
        return pvalue, None, None, None

    def make_sample(self, df: pd.DataFrame, metric_col: str = None):
        """
        Генерация подвыборки

        Parameters
        ----------
        df
        metric_col

        Returns
        -------

        """
        sample_size = int(min(self.max_n_sample, len(df) * self.percent_sample))
        if metric_col is not None:
            df = df[metric_col].sample(sample_size, replace=False).values
        else:
            df = df.sample(sample_size, replace=False)
        return df

    def compute_fpr(self, pvalues: list):
        """
        Доля pvalues, соответствующих условию проверки альфы

        Целевой результат – чтобы доля не превышала альфа

        Parameters
        ----------
        pvalues

        Returns
        -------

        """
        if len(pvalues) == 0:
            return None
        try:
            pvalues = np.array(pvalues)[~np.isnan(pvalues)]
            return float(float(sum(pvalues <= 1 - self.confidence_level) / len(pvalues)))
        except (ValueError, ZeroDivisionError) as e:
            logging.info(e)
            return None

    def stat_inference(
            self,
            df: pd.DataFrame,
            split_col: str,
            metric_col: str,
            estimator: str,
            metric: str,
            split_pair: list,
            post_power: bool = True

    ):
        """
        Расчет значимости для метрики

        Parameters
        ----------
        df : pd.DataFrame
        split_col : str
            Наименование колонки со сплитами
        metric_col : str
            Наименование числовой колонки с метрикой для проверки
        estimator : str
            Наименование стат. критерия {'t_test', 'prop', 'bootstrap', 'mann_whitney_test', 't_test_ratio'}
        metric : str
            Наименование метрики в конфиге
        post_power : bool
            Флаг расчета мощности

        Returns
        -------
        out : pd.DataFrame
            Результат расчета значимости
            - pvalue
            - is_significant

        """
        stat_res = {
            "split_pair": '_'.join(str(int(i)) for i in split_pair),
            "metric": metric,
            "pvalue": float,
            "power": None,
            "is_significant": bool,
            "delta": None,
            "lb": None,
            "ub": None,
        }

        estimators = {
            't_test': self.t_test,
            'mann_whitney_test': self.mann_whitney_test,
            't_test_ratio': self.t_test_ratio,
            'prop_test': self.prop_test,
            'bootstrap': self.bootstrap
        }

        split_x, split_y = [x for _, x in df.groupby(split_col)]

        if len(split_x) < 5 or len(split_y) < 5:
            pass
        if estimator == 't_test_ratio':
            stat_res['pvalue'], stat_res['delta'], stat_res['lb'], stat_res['ub'] = estimators[estimator](split_x,
                                                                                                          split_y)
        else:
            stat_res['pvalue'], stat_res['delta'], stat_res['lb'], stat_res['ub'] = estimators[estimator](
                split_x[metric_col],
                split_y[metric_col])

        stat_res['power'] = 0.0
        if post_power:
            res = self.fpr(split_x, split_y, metric_col, estimator)
            stat_res['power'] = res['fpr'][f'fpr_{config.CONFIDENCE_LEVEL}']
            res_df = pd.DataFrame({
                'pvalue': res['aa']['pvalue']
            })
            pr = '_'.join(str(int(i)) for i in split_pair)
            res_df['split_pair'] = pr
            res_df['metric'] = metric

        stat_res['is_significant'] = stat_res['pvalue'] < 1 - config.CONFIDENCE_LEVEL
        stat_res['is_significant'] = stat_res['is_significant'].astype('uint8')

        return pd.DataFrame([stat_res])

    def multiple_comparison(self, df: pd.DataFrame):
        """
        Множественная коррекция pvalue с сохранением cfg.CORRECTION_TYPE внутри эксперимента

        Parameters
        ----------
        df

        Returns
        -------
        out : pd.DataFrame
            Результаты коррекции
            - pvalue_adj
            - is_significant_adj
        """
        try:
            df['pvalue_adj'], df['is_significant_adj'] = pd.Series(dtype=float), pd.Series(dtype=int)
            df.loc[df['pvalue'].notna(), ['pvalue_adj']] = \
                multipletests(df['pvalue'][df['pvalue'].isna() == False],
                              alpha=1 - config.CONFIDENCE_LEVEL, method=config.CORRECTION_TYPE)[1]
            df['is_significant_adj'] = df['pvalue_adj'] < 1 - config.CONFIDENCE_LEVEL
            df['is_significant_adj'] = df['is_significant_adj'].astype(int)
        except ZeroDivisionError:
            df['is_significant_adj'] = df['pvalue']
        return df

