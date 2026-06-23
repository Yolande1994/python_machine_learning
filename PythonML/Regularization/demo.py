import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

# 造带噪声的数据
np.random.seed(42)
x = np.linspace(0, 10, 30)
y = 0.5 * x + np.sin(x) + np.random.randn(30) * 0.6
x_plot = np.linspace(0, 10, 200)

# 1. 高次多项式，一定会过拟合
model_overfit = make_pipeline(
    PolynomialFeatures(15),
    LinearRegression()
)
model_overfit.fit(x.reshape(-1,1), y)
y_overfit = model_overfit.predict(x_plot.reshape(-1,1))

# 2. 同样高次 + L2正则化（Ridge）= 变平滑
model_reg = make_pipeline(
    PolynomialFeatures(15),
    Ridge(alpha=1)  # 正则化强度
)
model_reg.fit(x.reshape(-1,1), y)
y_reg = model_reg.predict(x_plot.reshape(-1,1))

# 画图对比
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(x, y, c='r', label='数据')
plt.plot(x_plot, y_overfit, 'g-', lw=2, label='过拟合')
plt.title('不加正则化：疯狂拐弯')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(x, y, c='r', label='数据')
plt.plot(x_plot, y_reg, 'b-', lw=2, label='加了L2正则')
plt.title('加正则化：变平滑、学规律')
plt.legend()

plt.tight_layout()
plt.show()