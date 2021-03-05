# coding: utf-8

# DO NOT EDIT
# Autogenerated from the notebook predict.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # 预测（样本外）

import numpy as np
import statsmodels.api as sm

# ## 人工数据

nsample = 50
sig = 0.25
x1 = np.linspace(0, 20, nsample)
X = np.column_stack((x1, np.sin(x1), (x1 - 5)**2))
X = sm.add_constant(X)
beta = [5., 0.5, 0.5, -0.02]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

# ## 估计

olsmod = sm.OLS(y, X)
olsres = olsmod.fit()
print(olsres.summary())

# ## 样本内预测

ypred = olsres.predict(X)
print(ypred)

# ## 创建一个解释变量 Xnew 的新样本，进行预测和绘图

x1n = np.linspace(20.5, 25, 10)
Xnew = np.column_stack((x1n, np.sin(x1n), (x1n - 5)**2))
Xnew = sm.add_constant(Xnew)
ynewpred = olsres.predict(Xnew)  # predict out of sample
print(ynewpred)

# ## 绘图比较

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x1, y, 'o', label="Data")
ax.plot(x1, y_true, 'b-', label="True")
ax.plot(
    np.hstack((x1, x1n)),
    np.hstack((ypred, ynewpred)),
    'r',
    label="OLS prediction")
ax.legend(loc="best")

# ## 用公式预测

# 使用公式来估算和预测会更加简单。

from statsmodels.formula.api import ols

data = {"x1": x1, "y": y}

res = ols("y ~ x1 + np.sin(x1) + I((x1-5)**2)", data=data).fit()

# 我们使用 `I` 表示使用 Identity 转换。 即，我们不希望使用 `** 2` 来进行扩展

res.params

# 现在我们只需传入单个变量，就可以自动获取转换后的右侧变量

res.predict(exog=dict(x1=x1n))