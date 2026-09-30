import pandas as pd
import scipy
import matplotlib.pyplot as plt
all_data=pd.read_csv("C:\\Users\\Hmbb7\\Desktop\\kaggle\\train.csv")
all_data.info()
LotFrontage=all_data['LotFrontage']
LotFrontage=LotFrontage.fillna(LotFrontage.median())
sale_price=all_data['SalePrice']

line=scipy.stats.linregress(LotFrontage,sale_price)

#绘图
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 散点图
plt.scatter(LotFrontage, sale_price, alpha=0.4, color="steelblue", label="实际数据")

# 2. 拟合直线
x_line = LotFrontage                   # 用原始 x 范围即可
y_line = line.slope * x_line + line.intercept
plt.plot(x_line, y_line, color="red", linewidth=2,
         label=f"y = {line.slope:.1f}x + {line.intercept:.0f}")

# 3. 装饰
plt.xlabel("LotFrontage")
plt.ylabel("SalePrice")
plt.title(f"一元线性回归  R² = {line.rvalue**2:.3f}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
