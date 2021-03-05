# coding: utf-8

# DO NOT EDIT
# Autogenerated from the notebook formulas.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # 公式: 使用 R-style 公式拟合模型

# 从0.5.0版开始，statsmodels 允许用户使用 R-style 公式拟合统计模型。 在内部， ``statsmodels`` 使用 [patsy]包（http://patsy.readthedocs.org/）
# 将公式和数据转换为模型拟合中可使用的矩阵。 公式框架非常强大； 本教程仅涉及表面知识。 公式语言的完整说明可以在 ``patsy`` 文档中找到：
#
# * [Patsy 公式语言描述](http://patsy.readthedocs.org/)
#
# ## 加载模块和函数

import numpy as np  # noqa:F401  needed in namespace for patsy
import statsmodels.api as sm

# #### 导入规范

# 或者直接从 statsmodels.formula.api 导入 ols

from statsmodels.formula.api import ols

# 另外，您可以只使用主要 `statsmodels.api` 的 `formula` 命名空间。

sm.formula.ols

# 或者您可以使用以下规定

import statsmodels.formula.api as smf

# 这些名称只是访问每个模型的 `from_formula` 类方法的便捷方法。 参见，例如

sm.OLS.from_formula

# 所有小写的模型都接受 ``formula`` 和 ``data`` 参数，而大写的模型则采用 ``endog`` 和 ``exog`` 设计矩阵。 
# ``formula`` 接受一个以 ``patsy'' 公式描述模型的字符串。  ``data`` 接受[pandas]（https://pandas.pydata.org/）
# 数据框或为变量名称（如结构化数组或变量字典）定义 __getitem__ 的任何其他数据结构。

#
# ``dir(sm.formula)`` 将会输出可用的列表
#
# 公式兼容的模型具有以下通用呼叫签名: ``(formula, data, subset=None, *args, **kwargs)``

#
# ## 使用公式进行 OLS 回归
#
# 首先，我们拟合在[入门]（gettingstarted.html）页面上描述的线性模型。 下载数据，子集列和按列表删除以剔除缺少的观测值：


dta = sm.datasets.get_rdataset("Guerry", "HistData", cache=True)

df = dta.data[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
df.head()

# 拟合模型:

mod = ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
res = mod.fit()
print(res.summary())

# ## 分类变量
#
# 查看 summary 上面输出，请注意，``patsy`` 确定 *Region* 的元素是文本字符串，因此将 *Region* 视为分类变量。
# Patsy 的默认设置还包括截距，因此我们自动剔除 *Region* 分类的一个种族。
#
# 如果 *Region* 是我们视为类别的整数变量，则可以使用 ``C()`` 运算符来实现：


res = ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df).fit()
print(res.params)

# 在以下主题中讨论了 Patsy 分类变量的模式高级功能：[Patsy：分类变量的对比度编码系统]（contrasts.html）


# ## Operators
#
# 我们已经看到 "~" 将模型的左侧与右侧分开，而 "+" 将新列添加到设计矩阵中。
#
# ### 剔除变量
#
# "-" 符号可用于删除列/变量。 例如，我们可以通过以下方式从模型中删除截距：


res = ols(formula='Lottery ~ Literacy + Wealth + C(Region) -1 ', data=df).fit()
print(res.params)

# ### 乘法交互项
#
# ":" 通过其他两列的交互的新列添加到设计矩阵中。 "*" 还将包括相乘在一起的各个列：

res1 = ols(formula='Lottery ~ Literacy : Wealth - 1', data=df).fit()
res2 = ols(formula='Lottery ~ Literacy * Wealth - 1', data=df).fit()
print(res1.params, '\n')
print(res2.params)

# 运算符还可以实现许多其他功能。 请参阅 [patsy docs](https://patsy.readthedocs.org/en/latest/formulas.html) 了解跟多信息。

# ## 函数
#
# 您可以将向量化函数应用于模型中的变量:

res = smf.ols(formula='Lottery ~ np.log(Literacy)', data=df).fit()
print(res.params)

# 定义一个自定义函数:


def log_plus_1(x):
    return np.log(x) + 1.


res = smf.ols(formula='Lottery ~ log_plus_1(Literacy)', data=df).fit()
print(res.params)

# 公式中可以调用名称空间中的任何函数。

# ## 将公式应用于尚不支持它们的模型
#
# 即使给定的 `statsmodels` 函数不支持公式，您仍然可以使用 'patsy' 的公式语言来生成设计矩阵。
# 然后可以将这些矩阵作为 `endog` 和 `exog` 参数提供给拟合函数。
#
# 生成 ``numpy`` 数组:

import patsy
f = 'Lottery ~ Literacy * Wealth'
y, X = patsy.dmatrices(f, df, return_type='matrix')
print(y[:5])
print(X[:5])

# 生成 pandas 数据框:

f = 'Lottery ~ Literacy * Wealth'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
print(y[:5])
print(X[:5])

print(sm.OLS(y, X).fit().summary())