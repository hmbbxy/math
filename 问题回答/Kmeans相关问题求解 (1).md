# K-Means 相关问题求解

## 一、如何批量输出不同 K 值下「标签为 0」的样本

> 先纠正一个概念：K-Means 没有"不同迭代次数"这种说法。迭代次数（`max_iter`）是算法内部收敛步数，通常不遍历；你真正要遍历的是**簇数 K（`n_clusters`）**。下面按"不同 K 值"来解答。

+ 原代码与报错（你的截图）

<img width="2488" height="1426" alt="原代码与报错" src="https://github.com/user-attachments/assets/57604295-1272-4cac-a1d8-3ff51fef3913" />

+ 原代码问题
  - `labels` 是一个**列表**，每个元素是一次 KMeans 得到的标签数组。
  - 写成 `x_scale[labels==0]` 是错的：`labels==0` 是对整个列表做比较，不会按元素筛选，所以取不出样本。
  - 正确做法：取出单次标签数组 `lbl`，用 `lbl == 0` 生成布尔掩码去筛 `x_scale`。

+ 修正代码

```python
# 假设前面已经得到：
#   x_scale  —— 标准化后的 DataFrame（列 order_count / total_amount）
#   labels   —— 列表，labels[0] 对应 k=1，labels[1] 对应 k=2 ...
for k, lbl in enumerate(labels, start=1):
    mask = (lbl == 0)      # 哪些样本被分到了簇 0
    n0 = int(mask.sum())   # 标签为 0 的样本数
    sub = x_scale[mask]    # 这些样本本身
    print(f"k={k}  标签为0的样本数: {n0}")
    # 想看具体样本：print(sub) 或 sub.to_excel(f"label0_k{k}.xlsx")
```

> 说明：k=1 时所有样本都属于簇 0，所以 `n0 = 总样本数`，这是正常现象，不是 bug。

+ 运行结果（示例数据 240 条，仅供对照格式）

| K | 标签 0 样本数 | 总样本数 |
|---|---|---|
| 1 | 240 | 240 |
| 2 | 200 | 240 |
| 3 | 123 | 240 |
| 4 | 16  | 240 |
| 5 | 16  | 240 |
| 6 | 121 | 240 |
| 7 | 36  | 240 |
| 8 | 121 | 240 |
| 9 | 43  | 240 |

+ 图示（以 k=3 为例，红色为标签 0 的样本）

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# 1. 生成模拟数据 (为了演示，生成类似图中的数据分布)
# 这里生成 3 个簇，分别分布在左下、中间、右上
X, _ = make_blobs(n_samples=300, centers=[[-2, -2], [0, 0], [2, 2]], 
                  cluster_std=[0.5, 0.6, 0.8], random_state=42)

# 2. 训练 K-Means 模型 (K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_ # 获取质心坐标

# 3. 开始绘图
plt.figure(figsize=(10, 6), dpi=100) # 设置画布大小和清晰度

# --- 核心步骤：分别绘制不同的簇 ---
# 3.1 绘制簇 0 (高亮显示，红色)
# 注意：这里通过 labels == 0 进行布尔索引，筛选出属于簇0的数据点
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], 
            c='#E63946',  # 红色
            s=100,        # 点的大小
            alpha=0.9,    # 透明度
            label=f'簇0 (标签0, {np.sum(labels == 0)}个)') # 动态计算数量

# 3.2 绘制簇 1 和 簇 2 (灰度显示)
# 这里用循环遍历其他簇，统一设置为灰色
for i in range(1, 3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                c='#D3D3D3',  # 浅灰色
                s=100, 
                alpha=0.8, 
                label=f'簇{i}')

# 3.3 绘制质心 (深蓝色 X)
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='#1D3557',  # 深蓝色
            marker='X',   # X 形状
            s=300,        # 质心点要大一些
            edgecolors='white', # 边缘白色，增加立体感
            linewidths=1.5,
            label='质心',
            zorder=10)    # 确保质心画在最上层

# 4. 美化图表
plt.title('K=3 聚类结果：红色为标签0样本', fontsize=18, pad=20)
plt.xlabel('total_amount (标准化)', fontsize=14) # 根据你的业务修改
plt.ylabel('total_amount (标准化)', fontsize=14)

# 设置图例 (Legend)
plt.legend(fontsize=12, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

# 添加网格线
plt.grid(True, linestyle='-', alpha=0.3)

# 显示图表
plt.tight_layout()
plt.show()
```

![K=3 聚类：红色为标签0样本](kmeans_label0_demo.png)


## 二、如何批量输出轮廓系数

+ 原代码与报错（你的截图）

<img width="2508" height="334" alt="原输出" src="https://github.com/user-attachments/assets/3a81ea8f-b4df-448d-8510-f122fe1b2e35" />

+ 原代码问题
  - 那个警告来自 `plt.legend()`：画图时没给 `label`，legend 找不到内容，于是报 `No artists with labels found to put in legend`。修复：在 `plt.plot(..., label="SSE")`。
  - ⚠️ 轮廓系数部分：你算了 `All_score` 却没画出来；而且 SSE 和轮廓系数分成了两个循环，容易对不上。建议**合并到同一个循环**里同时算。
  - `silhouette_score` 要求 `2 ≤ 簇数 ≤ 样本数 - 1`，k=1 必须跳过（你已处理）。

+ 修正代码（合并版，直接可跑）

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1) 读数据 + 取特征
df = pd.read_excel(r"C:\Users\Hmbb7\Downloads\user_info.xlsx")
x = df[["order_count", "total_amount"]]

# 2) 标准化
scaler = StandardScaler()
x_scale = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)

# 3) 遍历 K=1..9，同时收集 SSE 与轮廓系数
sse, sil = [], []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=1, n_init=10)
    km.fit(x_scale)
    sse.append(km.inertia_)
    # 轮廓系数：簇数必须 2 <= k <= n-1
    sil.append(silhouette_score(x_scale, km.labels_) if 1 < k < len(x_scale) - 1 else np.nan)

# 4) 画图（都带 label，legend 不再报警告）
plt.figure()
plt.plot(range(1, 10), sse, "o-", color="#FB072F", label="SSE")
plt.xlabel("K 值"); plt.ylabel("SSE"); plt.legend(); plt.title("K 值 vs SSE（肘部法）")
plt.show()

ks = list(range(2, 10))
plt.figure()
plt.plot(ks, [sil[k - 1] for k in ks], "s-", color="#185FA5", label="轮廓系数")
plt.xlabel("K 值"); plt.ylabel("Silhouette"); plt.legend(); plt.title("K 值 vs 轮廓系数")
plt.show()

print("SSE      :", [round(v, 1) for v in sse])
print("轮廓系数 :", [round(v, 3) if v == v else None for v in sil])
```

+ 运行结果（示例数据，仅供对照格式）

```text
SSE      : [480.0, 124.8, 43.7, 33.2, 26.2, 21.0, 18.3, 16.3, 15.2]
轮廓系数 : [None, 0.726, 0.715, 0.676, 0.685, 0.585, 0.586, 0.57, 0.365]
```

+ 图示

![K 值 vs SSE 肘部图](kmeans_sse_elbow.png)

![K 值 vs 轮廓系数](kmeans_silhouette.png)

> 选 K 的口诀：**SSE 拐点（肘部）+ 轮廓系数最高点**，两者结合看。示例数据里 K=2~3 最稳。

---
*注：以上图片与数值均用与真实数据同结构的示例数据（`order_count` / `total_amount`）跑出，用于演示输出形态；把自己的 `user_info.xlsx` 代入上面的修正代码即可得到真实结果。复现脚本见同目录 `gen_kmeans.py`。*
