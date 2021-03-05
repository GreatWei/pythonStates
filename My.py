import pandas as pd
import statsmodels.api as sma
import matplotlib.pyplot as plt


inputExcel='winequality-both.csv'
wine=pd.read_csv(inputExcel)
wine.columns=wine.columns.str.replace(' ','_')
cols = ['type', 'quality']
#将wine的最后10行数据并剔除'tpye'字段和'quality'字段作为测试集
wineLast=wine.tail(10)
#生成自变量并添加常数项



def looper(limit):
    # 将wine第一行到倒数第11行的数据作为训练集,并为自变量加上常数项
    wineHead = wine.iloc[0:-10, :]
    # 生成自变量并添加常数项

    # 生成因变量
    wineTrainDep = wineHead['quality']



    for i in range(wine.columns.size):
        wineTrainInd = sma.add_constant(wineHead[wineHead.columns.difference(cols)])
        # 调用statsmodels模块的api.ols进行最小二乘线性回归
        lm = sma.OLS(wineTrainDep, wineTrainInd)
        # 生成回归结果
        result = lm.fit() #模型拟合
        pvalues = result.pvalues #得到结果中所有P值
        pvalues.drop('const',inplace=True) #把const取得
        pmax = max(pvalues) #选出最大的P值
        if pmax>limit:
            ind = pvalues.idxmax() #找出最大P值的index
            cols.append(ind) #把这个index从cols中删除

        else:
            return result
res = looper(0.05)
print(cols)

print(res.summary())

wineTest=sma.add_constant(wineLast[wineLast.columns.difference(cols)])
predict=res.predict(wineTest)
print("+++++++++++++++========")
print(predict)