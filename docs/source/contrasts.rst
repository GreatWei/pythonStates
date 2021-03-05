:orphan:

Patsy: 用于分类变量的对比编码系统
===========================================================

.. note:: 这个文档主要基于 `UCLA 的优秀资源  <http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm>`__.

K 个类别的分类变量通常以 K-1 个哑变量组的形式进入回归。这相当于水平均值的线性假设。也就是说，这些变量的每个检验统计量等同于检验该水平的平均值是否在统计上与基准类别的平均值显著不同。
这种伪编码在R术语中称为“处理编码”，我们将遵循此约定。但是，存在不同的编码方法，这些编码方法等于不同的线性假设集。

实际上，伪编码在技术上不是对比编码。这是因为伪变量加一，并且在功能上依赖于模型的截距。另一方面，具有 `k` 个级别的分类变量的一组 *contrasts* 是一组 `k-1` 个因子水平均值的函数独立线性组合，
它们也独立于哑变量之和。哑编码本身不是错误的。它捕获了所有系数，但是当模型假设系数独立时（例如在 ANOVA 中），会使问题复杂化。线性回归模型不假设系数的独立性，因此在这种情况下，通常仅采用虚拟编码。

要查看 Patsy 中的对比矩阵，我们将使用 UCLA ATS 的数据。首先，让我们加载数据。

.. ipython::
   :suppress:

   In [1]: import numpy as np
      ...: np.set_printoptions(precision=4, suppress=True)
      ...:
      ...: from patsy.contrasts import ContrastMatrix
      ...:
      ...: def _name_levels(prefix, levels):
      ...:     return ["[%s%s]" % (prefix, level) for level in levels]

   In [2]: class Simple(object):
      ...:     def _simple_contrast(self, levels):
      ...:         nlevels = len(levels)
      ...:         contr = -1./nlevels * np.ones((nlevels, nlevels-1))
      ...:         contr[1:][np.diag_indices(nlevels-1)] = (nlevels-1.)/nlevels
      ...:         return contr
      ...:
      ...:     def code_with_intercept(self, levels):
      ...:         contrast = np.column_stack((np.ones(len(levels)),
      ...:                                    self._simple_contrast(levels)))
      ...:         return ContrastMatrix(contrast, _name_levels("Simp.", levels))
      ...:
      ...:     def code_without_intercept(self, levels):
      ...:         contrast = self._simple_contrast(levels)
      ...:         return ContrastMatrix(contrast, _name_levels("Simp.", levels[:-1]))
      ...:

示例数据
------------

.. ipython:: python

   import pandas
   url = 'https://stats.idre.ucla.edu/stat/data/hsb2.csv'
   hsb2 = pandas.read_csv(url)

观测每个种族水平的因变量均值是有意义的（（1 =西班牙裔，2 =亚洲，3 =非裔美国人，4 =高加索人））。

.. ipython:: python

   hsb2.groupby('race')['write'].mean()

处理 (虚拟) 编码
------------------------

虚拟编码可能是最著名的编码方案。 它将分类变量的每个级别与基本参考级别进行比较。 基本参考级别是截距的值。 这是Patsy中无序分类因素的默认对比。 处理种族对比矩阵为

.. ipython:: python

   from patsy.contrasts import Treatment
   levels = [1,2,3,4]
   contrast = Treatment(reference=0).code_without_intercept(levels)
   print(contrast.matrix)

在这里，我们使用 `reference=0`，这意味着第一个级别（西班牙裔）是衡量其他级别影响的参考类别。 如上所述，
列的总和不为零，因此与截距无关。 明确地说，让我们看一下它如何编码 `race` 变量。

.. ipython:: python

   contrast.matrix[hsb2.race-1, :][:20]

这是一个技巧，因为种族类别可以方便地映射到 zero-based 的索引。 如果不这样做，那么这种转换将在幕后进行，
因此这通常不会起作用，但是仍然修正想法是有用的练习。 下面说明了使用上面三个对比的输出

.. ipython:: python

   from statsmodels.formula.api import ols
   mod = ols("write ~ C(race, Treatment)", data=hsb2)
   res = mod.fit()
   print(res.summary())

我们明确给出了种族的参照。 但是，由于 Treatment 是默认的，因此我们可以省略此设置。


简单编码
-------------

与处理编码类似，简单编码将每个级别与固定参考级别进行对比。 但是，使用简单编码，截距是所有因素水平的总和。有关如何实现简单对比的信息，请参见 :ref:`user-defined` 。



.. ipython:: python

   contrast = Simple().code_without_intercept(levels)
   print(contrast.matrix)

   mod = ols("write ~ C(race, Simple)", data=hsb2)
   res = mod.fit()
   print(res.summary())

求和 (Deviation) 编码
----------------------

求和编码将给定级别的因变量的平均值与所有级别上因变量的总体平均值进行比较。 也就是说，它使用了第一个 k-1 级别
和每个级别 k 之间进行对比。在此示例中，级别 1 与所有其他级别进行比较，级别 2 与所有其他级别进行比较，级别 3 与所有其他级别进行比较。

.. ipython:: python

   from patsy.contrasts import Sum
   contrast = Sum().code_without_intercept(levels)
   print(contrast.matrix)

   mod = ols("write ~ C(race, Sum)", data=hsb2)
   res = mod.fit()
   print(res.summary())

这与强制所有系数的总和为零的参数化相对应。 请注意，此处的截距是均值，其中均值是每个级别的因变量均值的均值。


.. ipython:: python

   hsb2.groupby('race')['write'].mean().mean()

后向差分编码
--------------------------

在后向差分编码中，将一个级别的因变量的平均值与先前级别的因变量的平均值进行比较。 这种类型的编码对于分类或有序变量可能很有用。


.. ipython:: python

   from patsy.contrasts import Diff
   contrast = Diff().code_without_intercept(levels)
   print(contrast.matrix)

   mod = ols("write ~ C(race, Diff)", data=hsb2)
   res = mod.fit()
   print(res.summary())

例如，此处关于级别 1 的系数是级别 2 的 `write` 的平均值，而不是级别 1 的平均值。


.. ipython:: python

   res.params["C(race, Diff)[D.1]"]
   hsb2.groupby('race').mean()["write"][2] - \
       hsb2.groupby('race').mean()["write"][1]

Helmert 编码
--------------

我们的 Helmert 编码版本有时被称为反向 Helmert 编码。 将某个级别的因变量的平均值与所有先前级别的因变量的平均值进行比较。
因此，有时使用 'reverse' 这个名称来区别于正向 Helmert 编码。 对于分类变量（例如 race），这种比较没有多大意义，但是我们将
使用 Helmert 对比，如下所示：


.. ipython:: python

   from patsy.contrasts import Helmert
   contrast = Helmert().code_without_intercept(levels)
   print(contrast.matrix)

   mod = ols("write ~ C(race, Helmert)", data=hsb2)
   res = mod.fit()
   print(res.summary())

为了说明这一点，第 4 级的参照是前三个级别的因变量的平均值，取第 4 级的平均值

.. ipython:: python

   grouped = hsb2.groupby('race')
   grouped.mean()["write"][4] - grouped.mean()["write"][:3].mean()

如您所见，这些仅等于一个常数。 Helmert对比的其他版本给出了实际的均值差异。 无论如何，假设检验是相同的。

.. ipython:: python

   k = 4
   1./k * (grouped.mean()["write"][k] - grouped.mean()["write"][:k-1].mean())
   k = 3
   1./k * (grouped.mean()["write"][k] - grouped.mean()["write"][:k-1].mean())


正交多项式编码
----------------------------

通过多项式编码对 `k=4` 级获得的系数，是分类变量的线性，二次和三次趋势。 此处假定分类变量由基本等距的数字变量表示。
因此，这种类型的编码仅用于具有相等间隔的有序分类变量。 通常，多项式对比产生阶数为 k-1 的多项式。 由于 `race` 不是
有序因子变量，因此我们以 `read` 为例。 首先，我们需要根据 `read` 创建一个有序的分类。


.. ipython:: python

   _, bins = np.histogram(hsb2.read, 3)
   try: # requires numpy master
       readcat = np.digitize(hsb2.read, bins, True)
   except:
       readcat = np.digitize(hsb2.read, bins)
   hsb2['readcat'] = readcat
   hsb2.groupby('readcat').mean()['write']

.. ipython:: python

   from patsy.contrasts import Poly
   levels = hsb2.readcat.unique().tolist()
   contrast = Poly().code_without_intercept(levels)
   print(contrast.matrix)

   mod = ols("write ~ C(readcat, Poly)", data=hsb2)
   res = mod.fit()
   print(res.summary())

如您所见，readcat 对因变量 `write` 有显著的线性影响，但对二次或三次无明显影响。


.. _user-defined:

用户自定义编码
-------------------

如果您想使用自定义的编码，则必须编写一个编码类，该类包含一个 code_with_intercept 和 code_without_intercept 方法，并返回一个 `patsy.contrast.ContrastMatrix` 实例。

.. ipython::

   In [1]: from patsy.contrasts import ContrastMatrix
      ...:
      ...: def _name_levels(prefix, levels):
      ...:     return ["[%s%s]" % (prefix, level) for level in levels]

   In [2]: class Simple(object):
      ...:     def _simple_contrast(self, levels):
      ...:         nlevels = len(levels)
      ...:         contr = -1./nlevels * np.ones((nlevels, nlevels-1))
      ...:         contr[1:][np.diag_indices(nlevels-1)] = (nlevels-1.)/nlevels
      ...:         return contr
      ...:
      ...:     def code_with_intercept(self, levels):
      ...:         contrast = np.column_stack((np.ones(len(levels)),
      ...:                                    self._simple_contrast(levels)))
      ...:         return ContrastMatrix(contrast, _name_levels("Simp.", levels))
      ...:
      ...:    def code_without_intercept(self, levels):
      ...:        contrast = self._simple_contrast(levels)
      ...:        return ContrastMatrix(contrast, _name_levels("Simp.", levels[:-1]))

   In [3]: mod = ols("write ~ C(race, Simple)", data=hsb2)
      ...: res = mod.fit()
      ...: print(res.summary())
