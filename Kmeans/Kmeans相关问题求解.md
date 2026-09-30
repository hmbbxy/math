# 如何批量的输出不同迭代次数下的标签为0的样本
+ 代码及报错

<img width="2488" height="1426" alt="00e226c80561b9b83dfb5f22dd766e2e" src="https://github.com/user-attachments/assets/57604295-1272-4cac-a1d8-3ff51fef3913" />

# 如何批量输出轮廓系数

+ 代码及报错

```python
import pandas as pd
import numpy as np

'''读取数据集并获取特征变量'''
# 读取文件，并赋值给变量df
df = pd.read_excel(r"C:\Users\Hmbb7\Downloads\user_info.xlsx")
df.info()
print(df.head())

# 获取特征变量x
x = df[["order_count","total_amount"]]



'''数据标准化'''
# 导入StandardScaler类
from sklearn.preprocessing import StandardScaler

# 创建一个StandardScaler对象
scaler = StandardScaler()

# 对x进行归一化(x' = (x - mean) / std)
x_scale = scaler.fit_transform(x)

print(x_scale)


# 将x_scale从二维数组转换为DataFrame
x_scale = pd.DataFrame(x_scale, columns=x.columns, index=x.index)

'''搭建K-Means模型'''
# 导入sklearn.cluster模块中的KMeans模型
from sklearn.cluster import KMeans

# 使用KMeans()初始化模型
# 设置参数n_clusters=i, random_state=1,寻找最佳k值
# 将结果赋值给model

sse=[]
labels=[]
for i in range(1,10):
    model = KMeans(n_clusters=i, random_state=1)

# 使用fit()函数训练模型
    model.fit(x_scale)
    sse.append(model.inertia_)
    labels.append(model.labels_)



print(f"标签是{labels}")
#看某一列具体有什么样本
###label_0=x_scale[labels==0]
###print(f"标签为0的样本为{label_0}")

'''⚠️⚠️轮廓系数'''
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples,silhouette_score

# 第二段代码改进：计算轮廓系数
All_score = []
K_range=range(1,10)
# 注意：这里应该使用训练模型时的数据 x_scale，而不是重新生成数据
for i, lbl in enumerate(labels):
    k = K_range[i] # 获取当前的 k 值
    if k > 1: # 只有当簇的数量大于1时，才计算轮廓系数
        # 确保标签长度和数据长度一致（通常是一致的，除非中间数据变了）
        if len(lbl) == len(x_scale):
            score = silhouette_score(x_scale, lbl)
            All_score.append(score)
        else:
            print(f"数据长度不匹配，跳过 k={k}")
            continue

print(All_score)




   
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.plot(range(1,10),sse,linewidth=2,color="#FB072F")
plt.xlabel("K值")
plt.ylabel("SSE")
plt.legend()
plt.title("K值vsSSE")
plt.show()
# 获取聚类后的质心
#print(f"质心：{model.cluster_centers_}")
# 获取每个样本所属的簇
#print(f"标签：{model.labels_}")
# 获取SSE
#print(f"SSE：{model.inertia_}")
```
+ 丢给ai改过几次，但还是报错（或者说是警告？），⚠️处应该是需要修改的部分
+ 部分输出：
```
标签是[array([0, 0, 0, ..., 0, 0, 0], shape=(51394,), dtype=int32), array([1, 1, 1, ..., 0, 0, 0], shape=(51394,), dtype=int32), array([2, 2, 2, ..., 0, 0, 0], shape=(51394,), dtype=int32), array([2, 2, 2, ..., 0, 0, 0], shape=(51394,), dtype=int32), array([4, 4, 4, ..., 0, 0, 0], shape=(51394,), dtype=int32), array([4, 4, 4, ..., 5, 5, 5], shape=(51394,), dtype=int32), array([4, 4, 4, ..., 5, 5, 5], shape=(51394,), dtype=int32), array([7, 7, 7, ..., 4, 4, 4], shape=(51394,), dtype=int32), array([7, 7, 7, ..., 4, 4, 4], shape=(51394,), dtype=int32)]
[0.7084912912255967, 0.6193451803848082, 0.5583749184931445, 0.5247688770789104, 0.4937205833926103, 0.49089062496113606, 0.47954408905174195, 0.46009324537928714]
c:\Users\Hmbb7\Desktop\数模备赛\规划\Kmean\K-mean.py:90: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  plt.legend()
```


