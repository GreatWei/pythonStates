# coding: utf-8

# DO NOT EDIT
# Autogenerated from the notebook recursive_ls.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # 递归最小二乘法
#
# 递归最小二乘法是普通最小二乘法的扩展窗口版本。 除了可用递归计算的回归系数之外，
# 递归计算的残差还可以构建统计量以研究参数的不稳定性。
#
# 
# `RecursiveLS` 类允许计算递归残差，并计算 CUSUM 和平方统计的 CUSUM。 将这些统计数据与参考线一起标出，
# 这些参考线表示与稳定参数的零假设在统计上的显著偏差，可以轻松直观地看到参数的稳定性。

# 最后，`RecursiveLS` 模型可以对参数向量施加线性限制，并且可以使用公式接口构建模型。


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader

np.set_printoptions(suppress=True)

# ## 例 1: Copper
#
# 我们首先考虑 Copper 数据集中的参数稳定性（如下所述）。

print(sm.datasets.copper.DESCRLONG)

dta = sm.datasets.copper.load_pandas().data
dta.index = pd.date_range('1951-01-01', '1975-01-01', freq='AS')
endog = dta['WORLDCONSUMPTION']

# 对于数据集中的回归变量，我们添加一列作为截距

exog = sm.add_constant(
    dta[['COPPERPRICE', 'INCOMEINDEX', 'ALUMPRICE', 'INVENTORYINDEX']])

# 首先，构建并生成模型，并输出 summary。 尽管 `RLS` 模型以递归方式计算回归参数，所以估计量与数据点一样多，
# 汇总表仅显示整个样本估计的回归参数。 除了来自递归初始化的细微影响外，这些估计值等同于OLS估计值。


mod = sm.RecursiveLS(endog, exog)
res = mod.fit()

print(res.summary())

# 递归系数可在 `recursive_coefficients` 属性中使用。或者，可以使用 `plot_recursive_coefficient` 方法生成图。


print(res.recursive_coefficients.filtered[0])
res.plot_recursive_coefficient(
    range(mod.k_exog), alpha=None, figsize=(10, 6))

# 可使用 `cusum` 属性来调用 CUSUM 统计量，但是通常使用 `plot_cusum` 方法可以更加方便直观地检查参数的稳定性。 
# 在下图中，CUSUM 统计量无法移除 5% 显著区间，因此我们无法拒绝 5% 水平下参数稳定性的原假设。


print(res.cusum)
fig = res.plot_cusum()

# 另一个相关的统计量是平方的 CUSUM，可使用 `cusum_squares` 属性来调用，但使用 `plot_cusum_squares` 方法可以
# 更加方便的进行可视化检查。 在下图中，平方统计的 CUSUM 无法移除 5% 的显著区间，因此我们无法拒绝 5% 水平下参数
# 稳定性的原假设。


res.plot_cusum_squares()

# # 例 2: 货币数量论
#
# 货币数量论认为，“货币数量变化率的给定变化会引起……价格通胀率的相等变化”(Lucas, 1980)。 在 Lucas 之后，我们研究了
# 货币增长的双向指数的加权移动平均值与 CPI 通胀之间的关系。 尽管卢卡斯发现这些变量之间的关系是稳定的，但最近看来这种
# 关系是不稳定的。 参见 Sargent 和 Surico（2010）示例。


start = '1959-12-01'
end = '2015-01-01'
m2 = DataReader('M2SL', 'fred', start=start, end=end)
cpi = DataReader('CPIAUCSL', 'fred', start=start, end=end)


def ewma(series, beta, n_window):
    nobs = len(series)
    scalar = (1 - beta) / (1 + beta)
    ma = []
    k = np.arange(n_window, 0, -1)
    weights = np.r_[beta**k, 1, beta**k[::-1]]
    for t in range(n_window, nobs - n_window):
        window = series.iloc[t - n_window:t + n_window + 1].values
        ma.append(scalar * np.sum(weights * window))
    return pd.Series(
        ma, name=series.name, index=series.iloc[n_window:-n_window].index)


m2_ewma = ewma(
    np.log(m2['M2SL'].resample('QS').mean()).diff().iloc[1:], 0.95, 10 * 4)
cpi_ewma = ewma(
    np.log(cpi['CPIAUCSL'].resample('QS').mean()).diff().iloc[1:], 0.95,
    10 * 4)

# 在使用 $\beta = 0.95$ Lucas 过滤器（每边有10年的窗口）构造移动平均值之后，我们在下面绘制每个序列。
# 尽管在部分样本中出现运行一致的情况，但在 1990 年之后出现了分歧。


fig, ax = plt.subplots(figsize=(13, 3))

ax.plot(m2_ewma, label='M2 Growth (EWMA)')
ax.plot(cpi_ewma, label='CPI Inflation (EWMA)')
ax.legend()

endog = cpi_ewma
exog = sm.add_constant(m2_ewma)
exog.columns = ['const', 'M2']

mod = sm.RecursiveLS(endog, exog)
res = mod.fit()

print(res.summary())

res.plot_recursive_coefficient(
    1, alpha=None)

# 现在，CUSUM 图在5% 的显著水平下展现了实质性偏差，表明拒绝了参数稳定性的零假设。

res.plot_cusum()

# 类似地，平方的 CUSUM 在 5%的显著水平下展现了实质性偏差，也表明拒绝了参数稳定性的零假设。

res.plot_cusum_squares()

# # 例 3: 线性限制和公式

# ### 线性限制
#
# 使用 `constraints` 参数构建模来实现线性限制并不难。


endog = dta['WORLDCONSUMPTION']
exog = sm.add_constant(
    dta[['COPPERPRICE', 'INCOMEINDEX', 'ALUMPRICE', 'INVENTORYINDEX']])

mod = sm.RecursiveLS(endog, exog, constraints='COPPERPRICE = ALUMPRICE')
res = mod.fit()
print(res.summary())

# ### 公式
#
# 可以使用类方法 `from_formula` 来拟合同一模型。

mod = sm.RecursiveLS.from_formula(
    'WORLDCONSUMPTION ~ COPPERPRICE + INCOMEINDEX + ALUMPRICE + INVENTORYINDEX',
    dta,
    constraints='COPPERPRICE = ALUMPRICE')
res = mod.fit()
print(res.summary())