# coding: utf-8

# DO NOT EDIT
# Autogenerated from the notebook interactions_anova.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # 交互作用 和 ANOVA

# 请注意: 这个脚本主要是基于 Jonathan Taylor 的课堂笔记
# http://www.stanford.edu/class/stats191/interactions.html
#
# 下载并格式化数据:

from statsmodels.compat import urlopen
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import pandas as pd
pd.set_option("display.width", 100)
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm

try:
    salary_table = pd.read_csv('salary.table')
except:  # 最近 pandas 可以在没有 urlopen 的情况下读取 URL
    url = 'http://stats191.stanford.edu/data/salary.table'
    fh = urlopen(url)
    salary_table = pd.read_table(fh)
    salary_table.to_csv('salary.table')

E = salary_table.E
M = salary_table.M
X = salary_table.X
S = salary_table.S

# 观察数据:

plt.figure(figsize=(6, 6))
symbols = ['D', '^']
colors = ['r', 'g', 'blue']
factor_groups = salary_table.groupby(['E', 'M'])
for values, group in factor_groups:
    i, j = values
    plt.scatter(
        group['X'], group['S'], marker=symbols[j], color=colors[i - 1], s=144)
plt.xlabel('Experience')
plt.ylabel('Salary')

# 拟合线性模型:

formula = 'S ~ C(E) + C(M) + X'
lm = ols(formula, salary_table).fit()
print(lm.summary())

# 查看创建的设计矩阵:

lm.model.exog[:5]

# 或者是由于我们最初传入了一个 DataFrame，所以会有一个 DataFrame 可用


lm.model.data.orig_exog[:5]

# 我们对原始不变的数据保持关注：

lm.model.data.frame[:5]

# 影响力分析

infl = lm.get_influence()
print(infl.summary_table())

# 或得到一个 DataFrame

df_infl = infl.summary_frame()

df_infl[:5]

# 现在分别绘制各组的组内残差：

resid = lm.resid
plt.figure(figsize=(6, 6))
for values, group in factor_groups:
    i, j = values
    group_num = i * 2 + j - 1  # 用于绘图
    x = [group_num] * len(group)
    plt.scatter(
        x,
        resid[group.index],
        marker=symbols[j],
        color=colors[i - 1],
        s=144,
        edgecolors='black')
plt.xlabel('Group')
plt.ylabel('Residuals')

# 现在我们使用 方差分析 或 f_检验 测量交互作用

interX_lm = ols("S ~ C(E) * X + C(M)", salary_table).fit()
print(interX_lm.summary())

# 方差分析检查

from statsmodels.stats.api import anova_lm

table1 = anova_lm(lm, interX_lm)
print(table1)

interM_lm = ols("S ~ X + C(E)*C(M)", data=salary_table).fit()
print(interM_lm.summary())

table2 = anova_lm(lm, interM_lm)
print(table2)

# 将设计矩阵设置为 DataFrame

interM_lm.model.data.orig_exog[:5]

# 将设计矩阵设置为 ndarray

interM_lm.model.exog
interM_lm.model.exog_names

infl = interM_lm.get_influence()
resid = infl.resid_studentized_internal
plt.figure(figsize=(6, 6))
for values, group in factor_groups:
    i, j = values
    idx = group.index
    plt.scatter(
        X[idx],
        resid[idx],
        marker=symbols[j],
        color=colors[i - 1],
        s=144,
        edgecolors='black')
plt.xlabel('X')
plt.ylabel('standardized resids')

# 看起来观察值是一个异常值。

drop_idx = abs(resid).argmax()
print(drop_idx)  # zero-based index
idx = salary_table.index.drop(drop_idx)

lm32 = ols('S ~ C(E) + X + C(M)', data=salary_table, subset=idx).fit()

print(lm32.summary())
print('\n')

interX_lm32 = ols('S ~ C(E) * X + C(M)', data=salary_table, subset=idx).fit()

print(interX_lm32.summary())
print('\n')

table3 = anova_lm(lm32, interX_lm32)
print(table3)
print('\n')

interM_lm32 = ols('S ~ X + C(E) * C(M)', data=salary_table, subset=idx).fit()

table4 = anova_lm(lm32, interM_lm32)
print(table4)
print('\n')

#  重绘残差

try:
    resid = interM_lm32.get_influence().summary_frame()['standard_resid']
except:
    resid = interM_lm32.get_influence().summary_frame()['standard_resid']

plt.figure(figsize=(6, 6))
for values, group in factor_groups:
    i, j = values
    idx = group.index
    plt.scatter(
        X[idx],
        resid[idx],
        marker=symbols[j],
        color=colors[i - 1],
        s=144,
        edgecolors='black')
plt.xlabel('X[~[32]]')
plt.ylabel('standardized resids')

#  绘制拟合值

lm_final = ols('S ~ X + C(E)*C(M)', data=salary_table.drop([drop_idx])).fit()
mf = lm_final.model.data.orig_exog
lstyle = ['-', '--']

plt.figure(figsize=(6, 6))
for values, group in factor_groups:
    i, j = values
    idx = group.index
    plt.scatter(
        X[idx],
        S[idx],
        marker=symbols[j],
        color=colors[i - 1],
        s=144,
        edgecolors='black')
    # 由于剔除了 NA 值，所以最终的模型中没有 idx 32 
    plt.plot(
        mf.X[idx].dropna(),
        lm_final.fittedvalues[idx].dropna(),
        ls=lstyle[j],
        color=colors[i - 1])
plt.xlabel('Experience')
plt.ylabel('Salary')

# 从我们的数据来看，管理组织的硕士学位和博士学位差异不同于非管理组织
# 这是两个定性变量管理（M）和教育（E）之间的交互作用，我们可以通过先
# 移除经验的影响，然后使用 interaction.plot 函数来绘制6组中每组的均值。


U = S - X * interX_lm32.params['X']

plt.figure(figsize=(6, 6))
interaction_plot(
    E,
    M,
    U,
    colors=['red', 'blue'],
    markers=['^', 'D'],
    markersize=10,
    ax=plt.gca())

# ## 少数族裔就业数据

try:
    jobtest_table = pd.read_table('jobtest.table')
except:  # 还没有数据
    url = 'http://stats191.stanford.edu/data/jobtest.table'
    jobtest_table = pd.read_table(url)

factor_group = jobtest_table.groupby(['MINORITY'])

fig, ax = plt.subplots(figsize=(6, 6))
colors = ['purple', 'green']
markers = ['o', 'v']
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)
ax.set_xlabel('TEST')
ax.set_ylabel('JPERF')

min_lm = ols('JPERF ~ TEST', data=jobtest_table).fit()
print(min_lm.summary())

fig, ax = plt.subplots(figsize=(6, 6))
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)

ax.set_xlabel('TEST')
ax.set_ylabel('JPERF')
fig = abline_plot(model_results=min_lm, ax=ax)

min_lm2 = ols('JPERF ~ TEST + TEST:MINORITY', data=jobtest_table).fit()

print(min_lm2.summary())

fig, ax = plt.subplots(figsize=(6, 6))
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)

fig = abline_plot(
    intercept=min_lm2.params['Intercept'],
    slope=min_lm2.params['TEST'],
    ax=ax,
    color='purple')
fig = abline_plot(
    intercept=min_lm2.params['Intercept'],
    slope=min_lm2.params['TEST'] + min_lm2.params['TEST:MINORITY'],
    ax=ax,
    color='green')

min_lm3 = ols('JPERF ~ TEST + MINORITY', data=jobtest_table).fit()
print(min_lm3.summary())

fig, ax = plt.subplots(figsize=(6, 6))
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)

fig = abline_plot(
    intercept=min_lm3.params['Intercept'],
    slope=min_lm3.params['TEST'],
    ax=ax,
    color='purple')
fig = abline_plot(
    intercept=min_lm3.params['Intercept'] + min_lm3.params['MINORITY'],
    slope=min_lm3.params['TEST'],
    ax=ax,
    color='green')

min_lm4 = ols('JPERF ~ TEST * MINORITY', data=jobtest_table).fit()
print(min_lm4.summary())

fig, ax = plt.subplots(figsize=(8, 6))
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)

fig = abline_plot(
    intercept=min_lm4.params['Intercept'],
    slope=min_lm4.params['TEST'],
    ax=ax,
    color='purple')
fig = abline_plot(
    intercept=min_lm4.params['Intercept'] + min_lm4.params['MINORITY'],
    slope=min_lm4.params['TEST'] + min_lm4.params['TEST:MINORITY'],
    ax=ax,
    color='green')

# MINORITY 对斜率或截距有什么影响？
table5 = anova_lm(min_lm, min_lm4)
print(table5)

# MINORITY 对截距 intercept 的影响
table6 = anova_lm(min_lm, min_lm3)
print(table6)

# MINORITY 对斜率 slope 的影响
table7 = anova_lm(min_lm, min_lm2)
print(table7)

# 只是对斜率有影响还是两者兼而有之？
table8 = anova_lm(min_lm2, min_lm4)
print(table8)

# ## 单尾 ANOVA

try:
    rehab_table = pd.read_csv('rehab.table')
except:
    url = 'http://stats191.stanford.edu/data/rehab.csv'
    rehab_table = pd.read_table(url, delimiter=",")
    rehab_table.to_csv('rehab.table')

fig, ax = plt.subplots(figsize=(8, 6))
fig = rehab_table.boxplot('Time', 'Fitness', ax=ax, grid=False)

rehab_lm = ols('Time ~ C(Fitness)', data=rehab_table).fit()
table9 = anova_lm(rehab_lm)
print(table9)

print(rehab_lm.model.data.orig_exog)

print(rehab_lm.summary())

# ## 双尾 ANOVA

try:
    kidney_table = pd.read_table('./kidney.table')
except:
    url = 'http://stats191.stanford.edu/data/kidney.table'
    kidney_table = pd.read_csv(url, delim_whitespace=True)

# 探索数据集

kidney_table.head(10)

# 平衡面板

kt = kidney_table
plt.figure(figsize=(8, 6))
fig = interaction_plot(
    kt['Weight'],
    kt['Duration'],
    np.log(kt['Days'] + 1),
    colors=['red', 'blue'],
    markers=['D', '^'],
    ms=10,
    ax=plt.gca())

# 您可以在公式评估名称空间中的调用名称空间并找到可用的东西

kidney_lm = ols('np.log(Days+1) ~ C(Duration) * C(Weight)', data=kt).fit()

table10 = anova_lm(kidney_lm)

print(
    anova_lm(
        ols('np.log(Days+1) ~ C(Duration) + C(Weight)', data=kt).fit(),
        kidney_lm))
print(
    anova_lm(
        ols('np.log(Days+1) ~ C(Duration)', data=kt).fit(),
        ols('np.log(Days+1) ~ C(Duration) + C(Weight, Sum)', data=kt).fit()))
print(
    anova_lm(
        ols('np.log(Days+1) ~ C(Weight)', data=kt).fit(),
        ols('np.log(Days+1) ~ C(Duration) + C(Weight, Sum)', data=kt).fit()))

# ## 平方和
#
# 说明了使用不同类型的平方和(I,II,III) 以及如何使用 Sum 对比度在3之间产生相同的输出。 
#
#  类型 I 和 II 在平衡设计下是等效的。
#
#  请勿使用非正交对比度的类型 III 即 Treatment

sum_lm = ols(
    'np.log(Days+1) ~ C(Duration, Sum) * C(Weight, Sum)', data=kt).fit()

print(anova_lm(sum_lm))
print(anova_lm(sum_lm, typ=2))
print(anova_lm(sum_lm, typ=3))

nosum_lm = ols(
    'np.log(Days+1) ~ C(Duration, Treatment) * C(Weight, Treatment)',
    data=kt).fit()
print(anova_lm(nosum_lm))
print(anova_lm(nosum_lm, typ=2))
print(anova_lm(nosum_lm, typ=3))