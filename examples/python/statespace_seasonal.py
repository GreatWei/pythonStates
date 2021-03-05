# coding: utf-8

# DO NOT EDIT
# Autogenerated from the notebook statespace_seasonal.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # 时间序列数据的季节性
#
# 考虑对多个具有不同周期性的季节性成分的时间序列数据进行建模的问题。让我们以时间序列 $y_t$ 进行分解，使其具有一个水平成分和两个季节性成分。
#
# $$
# y_t = \mu_t + \gamma^{(1)}_t + \gamma^{(2)}_t
# $$
#
# 其中 $\mu_t$ 表示趋势或水平，$\gamma^{(1)}_t$ 表示周期相对较短的季节性成分，而 $\gamma^{(2)}_t$ 表示较长时间的另一个季节性成分。
# 我们将为水平设置一个固定的截距项，并考虑 $\gamma^{(2)}_t$ 和 $\gamma^{(2)}_t$ 都是随机的，因此季节性模式会随时间变化。
# 
#
# 在这个笔记中，我们将生成符合这个模型的综合数据，并在未观察到的组件建模框架下以几种不同的方式展示季节性条件的建模。


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ### 创建合成数据
#
# 我们将按照 Durbin 和 Koopman（2012）中的方程（3.7）和（3.8）创建具有多个季节性模式的数据。 我们将模拟 300 个周期和
# 在频域中参数化的两个季节性项，分别具有周期 10 和 100，以及谐波数 3 和 2。 另外，它们的随机部分的方差分别是 4 和 9。



# 首先，我们将模拟合成数据
def simulate_seasonal_term(periodicity,
                           total_cycles,
                           noise_std=1.,
                           harmonics=None):
    duration = periodicity * total_cycles
    assert duration == int(duration)
    duration = int(duration)
    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    lambda_p = 2 * np.pi / float(periodicity)

    gamma_jt = noise_std * np.random.randn((harmonics))
    gamma_star_jt = noise_std * np.random.randn((harmonics))

    total_timesteps = 100 * duration  # Pad for burn in
    series = np.zeros(total_timesteps)
    for t in range(total_timesteps):
        gamma_jtp1 = np.zeros_like(gamma_jt)
        gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
        for j in range(1, harmonics + 1):
            cos_j = np.cos(lambda_p * j)
            sin_j = np.sin(lambda_p * j)
            gamma_jtp1[j - 1] = (
                gamma_jt[j - 1] * cos_j + gamma_star_jt[j - 1] * sin_j +
                noise_std * np.random.randn())
            gamma_star_jtp1[j - 1] = (
                -gamma_jt[j - 1] * sin_j + gamma_star_jt[j - 1] * cos_j +
                noise_std * np.random.randn())
        series[t] = np.sum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1
    wanted_series = series[-duration:]  # Discard burn in

    return wanted_series


duration = 100 * 3
periodicities = [10, 100]
num_harmonics = [3, 2]
std = np.array([2, 3])
np.random.seed(8678309)

terms = []
for ix, _ in enumerate(periodicities):
    s = simulate_seasonal_term(
        periodicities[ix],
        duration / periodicities[ix],
        harmonics=num_harmonics[ix],
        noise_std=std[ix])
    terms.append(s)
terms.append(np.ones_like(terms[0]) * 10.)
series = pd.Series(np.sum(terms, axis=0))
df = pd.DataFrame(data={
    'total': series,
    '10(3)': terms[0],
    '100(2)': terms[1],
    'level': terms[2]
})
h1, = plt.plot(df['total'])
h2, = plt.plot(df['10(3)'])
h3, = plt.plot(df['100(2)'])
h4, = plt.plot(df['level'])
plt.legend(['total', '10(3)', '100(2)', 'level'])
plt.show()

# ### 未观测到的组件 (频域模型)
#
# 下一种方法是未观测到的组件模型，其中将趋势性建模为固定截距，并使用具有一次周期分别为 10 和 100，谐波次数分别为 3 和 2 的三角函数
# 对季节性成分进行建模。请注意，这是正确的生成模型。时间序列的过程可以写成
#
# $$
# \begin{align}
# y_t & = \mu_t + \gamma^{(1)}_t + \gamma^{(2)}_t + \epsilon_t\\
# \mu_{t+1} & = \mu_t \\
# \gamma^{(1)}_{t} &= \sum_{j=1}^2 \gamma^{(1)}_{j, t} \\
# \gamma^{(2)}_{t} &= \sum_{j=1}^3 \gamma^{(2)}_{j, t}\\
# \gamma^{(1)}_{j, t+1} &= \gamma^{(1)}_{j, t}\cos(\lambda_j) + \gamma^{*,
# (1)}_{j, t}\sin(\lambda_j) + \omega^{(1)}_{j,t}, ~j = 1, 2, 3\\
# \gamma^{*, (1)}_{j, t+1} &= -\gamma^{(1)}_{j, t}\sin(\lambda_j) +
# \gamma^{*, (1)}_{j, t}\cos(\lambda_j) + \omega^{*, (1)}_{j, t}, ~j = 1, 2,
# 3\\
# \gamma^{(2)}_{j, t+1} &= \gamma^{(2)}_{j, t}\cos(\lambda_j) + \gamma^{*,
# (2)}_{j, t}\sin(\lambda_j) + \omega^{(2)}_{j,t}, ~j = 1, 2\\
# \gamma^{*, (2)}_{j, t+1} &= -\gamma^{(2)}_{j, t}\sin(\lambda_j) +
# \gamma^{*, (2)}_{j, t}\cos(\lambda_j) + \omega^{*, (2)}_{j, t}, ~j = 1,
# 2\\
# \end{align}
# $$
# $$
#
# 其中 $\epsilon_t$ 是白噪声， $\omega^{(1)}_{j,t}$ 是 i.i.d. $N(0, \sigma^2_1)$, 
# 而  $\omega^{(2)}_{j,t}$ 是 i.i.d. $N(0, \sigma^2_2)$,其中 $\sigma_1 = 2.$

model = sm.tsa.UnobservedComponents(
    series.values,
    level='fixed intercept',
    freq_seasonal=[{
        'period': 10,
        'harmonics': 3
    }, {
        'period': 100,
        'harmonics': 2
    }])
res_f = model.fit(disp=False)
print(res_f.summary())
# 第一个状态变量保存我们对截距的估计
print("fixed intercept estimated as {0:.3f}".format(
    res_f.smoother_results.smoothed_state[0, -1:][0]))

res_f.plot_components()
plt.show()

model.ssm.transition[:, :, 0]

# 观测到拟合的方差与真实方差 4 和 9 非常接近。此外，各个季节性成分看上去与真实的季节成分非常接近。平滑水平接近于真实水平 10 。
# 检验统计值是如此的小，无法拒绝我们的三个检验。


# ### 未观测到的组件（时域和频域混合建模）
#
# 第二种方法是未观测到的组件模型，其中将趋势建模为固定截距，使用 10 个常数之和为 0 和主周期为100，总谐波为2 的三角函数对季节性成分进行建模
# 请注意，这不是生成模型，因为它假定较短的季节性分量比实际存在更多的状态误差。时间序列的过程可以写成
#
# $$
# \begin{align}
# y_t & = \mu_t + \gamma^{(1)}_t + \gamma^{(2)}_t + \epsilon_t\\
# \mu_{t+1} & = \mu_t \\
# \gamma^{(1)}_{t + 1} &= - \sum_{j=1}^9 \gamma^{(1)}_{t + 1 - j} +
# \omega^{(1)}_t\\
# \gamma^{(2)}_{j, t+1} &= \gamma^{(2)}_{j, t}\cos(\lambda_j) + \gamma^{*,
# (2)}_{j, t}\sin(\lambda_j) + \omega^{(2)}_{j,t}, ~j = 1, 2\\
# \gamma^{*, (2)}_{j, t+1} &= -\gamma^{(2)}_{j, t}\sin(\lambda_j) +
# \gamma^{*, (2)}_{j, t}\cos(\lambda_j) + \omega^{*, (2)}_{j, t}, ~j = 1,
# 2\\
# \end{align}
# $$
#
# 其中 $\epsilon_t$ 是白噪声, $\omega^{(1)}_{t}$ 是 i.i.d. $N(0, \sigma^2_1)$, 且 $\omega^{(2)}_{j,t}$ 是 i.i.d. $N(0, \sigma^2_2)$。

model = sm.tsa.UnobservedComponents(
    series,
    level='fixed intercept',
    seasonal=10,
    freq_seasonal=[{
        'period': 100,
        'harmonics': 2
    }])
res_tf = model.fit(disp=False)
print(res_tf.summary())
# 第一个状态变量保存我们对截距的估计
print("fixed intercept estimated as {0:.3f}".format(
    res_tf.smoother_results.smoothed_state[0, -1:][0]))

res_tf.plot_components()
plt.show()

# 绘制的组件看起来不错。然而，第二个季节性的估计方差比真实的更加膨大。此外，我们拒绝了 Ljung-Box 统计量，
# 这表明在考虑了我们的组成成分之后，可能仍然存在自相关的问题。


# ### 未观测到的组件 (懒惰频域建模)
#
# 第三种方法是具有固定截距和一个季节性成分的未观测到的组件模型，该模型是使用具有 100 和 50 谐波的三角函数来建模。 
# 请注意，这不是生成模型，因为它假定实际上存在更多的谐波。由于方差联系在一起，因此我们无法将不存在的谐波的估计协方差驱动设置为0。
# 此模型规范的懒惰是，我们不必费心指定两个不同的季节性成分，而是选择建模它们使用具有足够谐波的单个分量来覆盖两者。 我们将无法捕获
# 两个真实成分之间方差的任何差异。时间序列进程可以写成：
#
# $$
# \begin{align}
# y_t & = \mu_t + \gamma^{(1)}_t + \epsilon_t\\
# \mu_{t+1} &= \mu_t\\
# \gamma^{(1)}_{t} &= \sum_{j=1}^{50}\gamma^{(1)}_{j, t}\\
# \gamma^{(1)}_{j, t+1} &= \gamma^{(1)}_{j, t}\cos(\lambda_j) + \gamma^{*,
# (1)}_{j, t}\sin(\lambda_j) + \omega^{(1}_{j,t}, ~j = 1, 2, \dots, 50\\
# \gamma^{*, (1)}_{j, t+1} &= -\gamma^{(1)}_{j, t}\sin(\lambda_j) +
# \gamma^{*, (1)}_{j, t}\cos(\lambda_j) + \omega^{*, (1)}_{j, t}, ~j = 1, 2,
# \dots, 50\\
# \end{align}
# $$
#
# 其中 $\epsilon_t$ 是白噪声, $\omega^{(1)}_{t}$ 是 i.i.d. $N(0, \sigma^2_1)$.

model = sm.tsa.UnobservedComponents(
    series, level='fixed intercept', freq_seasonal=[{
        'period': 100
    }])
res_lf = model.fit(disp=False)
print(res_lf.summary())
# 第一个状态变量保存我们对截距的估计
print("fixed intercept estimated as {0:.3f}".format(
    res_lf.smoother_results.smoothed_state[0, -1:][0]))

res_lf.plot_components()
plt.show()

#请注意，我们的诊断检验之一将在 0.05 水平下被拒绝。

# ### 未观测到的组件(懒惰时域季节性建模)
#
# 第四种方法是具有固定截距和使用 100 个常数的时域季节性模型建模的单个季节性成分的未观测到的组件模型。时间序列的过程可以写成：
#
# $$
# \begin{align}
# y_t & =\mu_t + \gamma^{(1)}_t + \epsilon_t\\
# \mu_{t+1} &= \mu_{t} \\
# \gamma^{(1)}_{t + 1} &= - \sum_{j=1}^{99} \gamma^{(1)}_{t + 1 - j} +
# \omega^{(1)}_t\\
# \end{align}
# $$
#
# 其中 $\epsilon_t$ 是白噪声, $\omega^{(1)}_{t}$ 是 i.i.d. $N(0, \sigma^2_1)$.

model = sm.tsa.UnobservedComponents(
    series, level='fixed intercept', seasonal=100)
res_lt = model.fit(disp=False)
print(res_lt.summary())
# 第一个状态变量保存我们对截距的估计
print("fixed intercept estimated as {0:.3f}".format(
    res_lt.smoother_results.smoothed_state[0, -1:][0]))

res_lt.plot_components()
plt.show()

# 季节性成分本身看起来不错——这是主信号。 季节性的估计方差非常高（$> 10 ^ 5 $），导致我们的提前 one-step 预测存在很大不确定性，
# 并且对新数据的响应速度较慢，这一点可通过提前 one-step 预测和观测的较大误差来证明。最终，我们的所有三个诊断检验均被拒绝。


# ### 滤波估算值比较
# 
# 下图显示了在滤波状态下在大约半个周期内对各个成分进行建模会接近真实状态。懒惰模型花费了更长的时间（几乎是整个周期）来对组合的真实状态执行相同的操作。


# 为季节性指定更好的名称
true_seasonal_10_3 = terms[0]
true_seasonal_100_2 = terms[1]
true_sum = true_seasonal_10_3 + true_seasonal_100_2

time_s = np.s_[:50]  # 在此之后，他们基本上同意
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
h1, = ax1.plot(
    series.index[time_s],
    res_f.freq_seasonal[0].filtered[time_s],
    label='Double Freq. Seas')
h2, = ax1.plot(
    series.index[time_s],
    res_tf.seasonal.filtered[time_s],
    label='Mixed Domain Seas')
h3, = ax1.plot(
    series.index[time_s],
    true_seasonal_10_3[time_s],
    label='True Seasonal 10(3)')
plt.legend(
    [h1, h2, h3], ['Double Freq. Seasonal', 'Mixed Domain Seasonal', 'Truth'],
    loc=2)
plt.title('Seasonal 10(3) component')
plt.show()

time_s = np.s_[:50]  # 在此之后，他们基本上同意
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
h21, = ax2.plot(
    series.index[time_s],
    res_f.freq_seasonal[1].filtered[time_s],
    label='Double Freq. Seas')
h22, = ax2.plot(
    series.index[time_s],
    res_tf.freq_seasonal[0].filtered[time_s],
    label='Mixed Domain Seas')
h23, = ax2.plot(
    series.index[time_s],
    true_seasonal_100_2[time_s],
    label='True Seasonal 100(2)')
plt.legend(
    [h21, h22, h23],
    ['Double Freq. Seasonal', 'Mixed Domain Seasonal', 'Truth'],
    loc=2)
plt.title('Seasonal 100(2) component')
plt.show()

time_s = np.s_[:100]

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
h31, = ax3.plot(
    series.index[time_s],
    res_f.freq_seasonal[1].filtered[time_s] +
    res_f.freq_seasonal[0].filtered[time_s],
    label='Double Freq. Seas')
h32, = ax3.plot(
    series.index[time_s],
    res_tf.freq_seasonal[0].filtered[time_s] +
    res_tf.seasonal.filtered[time_s],
    label='Mixed Domain Seas')
h33, = ax3.plot(
    series.index[time_s], true_sum[time_s], label='True Seasonal 100(2)')
h34, = ax3.plot(
    series.index[time_s],
    res_lf.freq_seasonal[0].filtered[time_s],
    label='Lazy Freq. Seas')
h35, = ax3.plot(
    series.index[time_s],
    res_lt.seasonal.filtered[time_s],
    label='Lazy Time Seas')

plt.legend(
    [h31, h32, h33, h34, h35], [
        'Double Freq. Seasonal', 'Mixed Domain Seasonal', 'Truth',
        'Lazy Freq. Seas', 'Lazy Time Seas'
    ],
    loc=1)
plt.title('Seasonal components combined')
plt.show()

# ##### 结论
#
# 在这个笔记中，我们模拟了一个具有两个不同时期的季节性成分的时间序列。我们使用结构时间序列模型对它们进行建模，其中：
#   （a）具有正确周期和谐波数量的两个频域成分；
#   （b）具有正确周期和谐波数量的频域项和更短周期的两个频域成分；
#   （c）具有较长周期和全部谐波的单个频域项；
#   （d）具有较长周期的单个时域项。
# 我们看到了各种各样的诊断结果，只有正确的生成模型（a）才能拒绝任何检验。因此，更灵活的季节性建模可允许使用多个具有特定谐波的成分，
# 这对于时间序列建模是有用的工具。最后，我们可以用这种方式表示总状态较少的季节性成分，从而允许用户尝试自己进行偏差方差的权衡，而不是
# 被迫选择 "lazy" 模型——使用大量状态并导致额外差异作为结果。