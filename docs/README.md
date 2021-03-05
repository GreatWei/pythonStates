# 文档的文档

我们的文档使用了一系列的 sphinx 文档和 jupyter notebooks。Jupyter notebooks 用于展示一个更长的、独立的示例。
Sphinx 文档也很不错，对于目录表和API文档更加合适。

## 建立的过程

构建文档需要一些其他的依赖关系。您可以通过以下方式获得大多数的依赖关系文档

```bash

   pip install -e .[docs]

```

从项目的根源开始，一些示例依赖于 rpy2 来执行 R 代码文档，由于难以安装，因此它并不包含在安装要求中。

要生成 HTML 文档, 从docs目录中运行 make html，这将执行以下类型的文件

1. datasets
2. notebooks
3. sphinx

# 构建 Notebook

我们用 nbconvert 工具来运行 notebook，然后将它们转换为HTML，交由 statsmodels/tools/nbgenerate.py 处理。默认的 python内核（嵌入在笔记本中）为 python3。您至少需要通过nbconvert==4.2.0 指定非默认内核，该内核可以在Makefile中传递。
