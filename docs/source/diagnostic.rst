:orphan:

.. _diagnostics:

回归诊断和设定性检验
==============================================


介绍
------------

在许多统计分析示例中，我们不确定统计模型是否正确指定，例如，当使用 ols 时，假定线性和方差齐，另外一些统计检验则假定误差服从正态分布的，或者我们有大量样本。
由于我们得到的统计结果依赖于这些统计假设，因此结果仅在我们的假设成立（至少近似）时才是正确的。

正确地解决不确定性问题的方法是使用鲁棒性的方法，例如，鲁棒性回归或鲁棒性（sandwich）协方差估计。
第二种方法是检验我们的样本是否服从这些假设。

以下简要概述了线性回归的回归诊断和假设检验

异方差检验
------------------------

对于这些检验，零假设是所有观测值都具有相同的误差方差，即残差齐。检验的不同之处在于，统计检验差异是哪种异方差被视为替代假设。
对于不同类型的异方差性，检验权重也会有所不同。

:py:func:`het_breuschpagan <statsmodels.stats.diagnostic.het_breuschpagan>`
    Breusch-Pagan 的拉格朗日乘数法的异方差检验

:py:func:`het_white <statsmodels.stats.diagnostic.het_white>`
    White 的拉格朗日乘数法的异方差检验

:py:func:`het_goldfeldquandt <statsmodels.stats.diagnostic.het_goldfeldquandt>`
    检验 2 个子样本中的方差是否相同


自相关检验
---------------------

这组测试回归残差是否自相关的。他们假定观测值是按时间排序的。

:py:func:`durbin_watson <statsmodels.stats.diagnostic.durbin_watson>`
  - Durbin-Watson 检验，残差无自相关性
  - 以 summary() 方法输出

:py:func:`acorr_ljungbox <statsmodels.stats.diagnostic.acorr_ljungbox>`
  - Ljung-Box 检验，残差无自相关性
  - 还返回 Box-Pierce 统计量

:py:func:`acorr_breusch_godfrey <statsmodels.stats.diagnostic.acorr_breusch_godfrey>`
  - Breusch-Pagan 检验，残差无自相关性


missing
  - ?


非线性检验
-------------------

:py:func:`linear_harvey_collier <statsmodels.stats.diagnostic.linear_harvey_collier>`
  - 线性规范是正确的零假设的乘数检验

:py:func:`acorr_linear_rainbow <statsmodels.stats.diagnostic.acorr_linear_rainbow>`
  - 线性规范是正确的零假设的乘数检验

:py:func:`acorr_linear_lm <statsmodels.stats.diagnostic.acorr_linear_lm>`
  - 拉格朗日乘数法检验是用于线性规范是正确的零假设，这种检验反对特定功能的替代。

:py:func:`spec_white <statsmodels.stats.diagnostic.spec_white>`
  - White 的 two-moment 规范检验，具有同方差的零假设并正确指定。

结构变化、参数稳定性检验
------------------------------------------------

检验全部或某些回归系数在整个数据样本中是否恒定。

已知变更点
^^^^^^^^^^^^^^^^^^

OneWayLS :
  - 弹性 ols 装饰器，用于预定义的子样本（eg. 组）的检验相同回归系数

missing
  - 预测检验：Greene，子样本中的观察数小于回归数


未知变更点
^^^^^^^^^^^^^^^^^^^^

:py:func:`breaks_cusumolsresid <statsmodels.stats.diagnostic.breaks_cusumolsresid>`
  - 参数稳定性的 cusum 检验是基于 ols 残差

:py:func:`breaks_hansen <statsmodels.stats.diagnostic.breaks_hansen>`
  - 模型稳定性检验，刷新 ols 参数，Hansen 1992

:py:func:`recursive_olsresiduals <statsmodels.stats.diagnostic.recursive_olsresiduals>`
  计算带有残差和 cusum 检验统计的递归 ols。当前，关于递归残差的辅助函数主要基于检验。但是，由于它使用递归更新并且不估计单独的问题，
  因此在扩展 OLS 功能时也应该非常有效。

missing
  - supLM, expLM, aveLM  (Andrews, Andrews/Ploberger)
  - R-结构变化也有 musum (移动累积总和检验)
  - 有哪些递归参数估计的检验


多重共线性检验
--------------------------------

conditionnum (statsmodels.stattools)
  - -- needs test vs Stata --
  - cf Grene (3rd ed.) pp 57-8

numpy.linalg.cond
  - (获取更多一般条件编号，但无助于幕后的设计准备)

方差膨胀因素
  目前，这与影响力和异常值措施一起 (此处有一些其他检验的链接: http://www.stata.com/help.cgi?vif)


正态分布检验
--------------------------------

:py:func:`jarque_bera <statsmodels.stats.tools.jarque_bera>`
  - 以 summary() 输出
  - 残差正态性检验

科学统计中的正态性检验
  需要再次找到清单

:py:func:`omni_normtest <statsmodels.stats.tools.omni_normtest>`
  - 检验残差的正态分布
  - 以 summary() 输出

:py:func:`normal_ad <statsmodels.stats.diagnostic.normal_ad>`
  - Anderson Darling 检验均值和方差的正态性
:py:func:`kstest_normal <statsmodels.stats.diagnostic.kstest_normal>` :py:func:`lilliefors <statsmodels.stats.diagnostic.lilliefors>`
  Lilliefors 正态性检验，这是一个关于均值和方差的 Kolmogorov-Smirnov 检验，lilliefors 是 kstest_normal 的别名

qqplot, scipy.stats.probplot

scipy.stats 的分布和增强功能是其他拟合优度检验
other goodness-of-fit tests for distributions in scipy.stats and enhancements
  - kolmogorov-smirnov
  - anderson : Anderson-Darling
  - 似然比, ...
  - 卡方检验，功效差异: 需要封装(for binning)


异常值和影响的诊断措施
-----------------------------------------

这些措施试图确定离群值较大，残差较大的观测值或对回归估计值影响较大的观测值。稳健回归RLM
可用于以异常健壮的方式进行估计以及识别异常。RLM的优点是，即使存在许多异常值，估计结果也
不会受到很大的影响，而大多数其他措施则可以更好地识别单个异常值，并且可能无法识别异常值组。

:py:class:`RLM <statsmodels.robust.robust_linear_model.RLM>`
    示例来自 example_rlm.py ::

        import statsmodels.api as sm

        ### 在默认的中位数绝对偏差标度下使用 Huber 的 T范数的示例
       

        data = sm.datasets.stackloss.load()
        data.exog = sm.add_constant(data.exog)
        huber_t = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
        hub_results = huber_t.fit()
        print(hub_results.weights)

    权重给出了根据要求的缩放比例将特定观察权降低多少的想法。
  

:py:class:`Influence <statsmodels.stats.outliers_influence.OLSInfluence>`
   stats.outliers_influence 中的类, 对于离群值和影响力的大多数标准度量都可以作为给定的 OLS 模型提供的方法或属性来使用。 
   这主要是为OLS编写的，某些（但不是全部）量度对其他模型也有效。这些统计信息中的一些可以从OLS结果实例计算得出，其他一些则需要为每个遗漏变量估算OLS。

   - resid_press
   - resid_studentized_external
   - resid_studentized_internal
   - ess_press
   - hat_matrix_diag
   - cooks_distance - Cook's Distance `Wikipedia <https://en.wikipedia.org/wiki/Cook%27s_distance>`_ (with some other links)
   - cov_ratio
   - dfbetas
   - dffits
   - dffits_internal
   - det_cov_params_not_obsi
   - params_not_obsi
   - sigma2_not_obsi



单位根检验
---------------

:py:func:`unitroot_adf <statsmodels.stats.diagnostic.unitroot_adf>`
  - 与 adfuller 相同，但签名不同