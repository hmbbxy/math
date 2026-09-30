import numpy as np

def ahp_geometric_method(matrix):
    """方根法(几何平均法)计算AHP权重"""
    n = matrix.shape[0]
    # 1. 计算每行几何平均
    row_geometric_mean = np.prod(matrix, axis=1) ** (1/n)
    # 2. 归一化得到权重
    weights = row_geometric_mean / np.sum(row_geometric_mean)
    return weights

def ahp_sum_product_method(matrix):
    """和积法计算AHP权重"""
    n = matrix.shape[0]
    # 1. 按列归一化
    col_sum = np.sum(matrix, axis=0)
    normalized_matrix = matrix / col_sum
    # 2. 按行求和
    row_sum = np.sum(normalized_matrix, axis=1)
    # 3. 再次归一化
    weights = row_sum / np.sum(row_sum)
    return weights

def consistency_check(matrix, weights):
    """一致性检验"""
    n = matrix.shape[0]
    # 计算最大特征值 lambda_max
    # 对于方根法，lambda_max ≈ sum((A*w)_i / w_i) / n
    Aw = np.dot(matrix, weights)
    lambda_max = np.mean(Aw / weights)
    
    CI = (lambda_max - n) / (n - 1)
    
    # 平均随机一致性指标 RI (1-10阶)
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
               6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49) # 若n>10，用1.49近似
    
    CR = CI / RI if RI != 0 else 0
    return CR

#灵敏度测试
#扰动准则层权重（倍数调动）
def disturb_weight_scale(origin_w,idx,scale):
    w_new=origin_w.copy()
    w_new[idx]=origin_w[idx]*scale
    res_sum=np.sum(w_new)-w_new[idx]
    for i in range(len((origin_w))):
        if i !=idx:
            res_ratio=origin_w[i]/np.sum(origin_w)
            w_new[i]=res_sum*res_ratio
    return w_new

#自定义调动
def disturb_weight_figure(origin_w,idx,figure):
    w_new=origin_w.copy()
    w_new[idx]=figure
    res_sum=np.sum(w_new)-w_new[idx]
    for i in range(len((origin_w))):
        if i !=idx:
            res_ratio=origin_w[i]/np.sum(origin_w)
            w_new[i]=res_sum*res_ratio
    return w_new








# 准则层判断矩阵 (质量 vs 成本 vs 效率)
# 假设: 质量比成本稍重要(3), 质量比效率明显重要(5), 成本比效率稍重要(2)
A_criteria = np.array([
    [1,   3,   5],
    [1/3, 1,   2],
    [1/5, 1/2, 1]
])
#excel

print("=== 准则层权重计算 ===")
w_criteria_geo = ahp_geometric_method(A_criteria)
print(w_criteria_geo)
w_criteria_sum = ahp_sum_product_method(A_criteria)
print(w_criteria_sum)
#各准则及对应权重
#柱形图描绘
name=['质量','成本','效率']
import matplotlib.pyplot  as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
#柱形图
plt.figure(figsize=(8,6),dpi=100)
plt.bar(name,w_criteria_geo)
plt.title('准则层对应指标的权重')
plt.xlabel('准则层指标')
plt.ylabel('权重')
#plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.grid(True,alpha=0.3)
plt.savefig('准则.pdf')


print(f"方根法(几何平均)权重: {w_criteria_geo}")
print(f"和积法权重: {w_criteria_sum}")

CR = consistency_check(A_criteria, w_criteria_geo)
print(f"一致性比率 CR = {CR:.4f} {'✅ 通过' if CR < 0.1 else '❌ 未通过'}")

# 假设方案层对准则的权重已知（此处省略计算过程）
# 方案层权重 (供应商A, B, C) 对各个准则
# 例如：对质量准则的方案权重
w_suppliers_quality = np.array([0.6, 0.3, 0.1])
w_suppliers_cost = np.array([0.2, 0.5, 0.3])
w_suppliers_efficiency = np.array([0.3, 0.2, 0.5])
'''单个玫瑰图
# 1. 先设样式和字体
plt.style.use("ggplot")
plt.rcParams["font.sans-serif"] = ["SimHei"]      # 中文字体
plt.rcParams["axes.unicode_minus"] = False        # 负号正常显示

# 2. 数据
value = ['供应商A', '供应商B', '供应商C']

N = len(w_suppliers_quality)

theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
width = 2 * np.pi / N

# 3. 自定义颜色
colors = ['#B82D16', '#047011', '#0F94FA']

# 4. 画布（样式设完再建）
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

# 5. 画玫瑰图
ax.bar(theta, w_suppliers_quality, width=width, bottom=0,
       color=colors[:N],          # 按柱子数取色
       alpha=0.7, edgecolor="white")

# 6. 角度刻度设为类别名
ax.set_xticks(theta)
ax.set_xticklabels(value)
ax.set_title("对质量准则的权重")

plt.show()
'''

#多个玫瑰图
data_list=[([0.6, 0.3, 0.1],['供应商A','供应商B','供应商C']),([0.2, 0.5, 0.3],['供应商A','供应商B','供应商C']),([0.3, 0.2, 0.5],['供应商A','供应商B','供应商C'])]
titles=['对质量准则的权重','对成本准则的权重','对效率准则的权重']
colors=['#B82D16','#047011','#0F94FA']
fig,axes=plt.subplots(1,3,figsize=(10,10),
                      subplot_kw={'projection':"polar"})
for ax,(data,labels),title in zip(axes.flat,data_list,titles):
    N=len(data)
    theta=np.linspace(0,2*np.pi,N,endpoint=False)
    width=2*np.pi/N
    ax.bar(theta,data,width=width,bottom=0,color=colors[:N],alpha=0.7,edgecolor='white')
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('对准则层各个指标的权重.pdf')
# 层次总排序
# 最终得分 = 质量权重 * 方案对质量权重 + 成本权重 * 方案对成本权重 + 效率权重 * 方案对效率权重
final_scores = (w_criteria_geo[0] * w_suppliers_quality + 
                w_criteria_geo[1] * w_suppliers_cost + 
                w_criteria_geo[2] * w_suppliers_efficiency)

#折线图
GS_name=['供应商A','供应商B','供应商C']
plt.figure(figsize=(8,5))
plt.plot(GS_name,final_scores,color='#0F94FA',linewidth=2,linestyle='-',marker='o',markersize=6,label='得分')
plt.title('层次总排序')
plt.xlabel('供应商')
plt.ylabel('最终得分')
plt.legend()
#plt.grid(True,alpha=0.5)
plt.savefig('最终得分.pdf')


print("\n=== 层次总排序结果 ===")
suppliers = ['供应商A', '供应商B', '供应商C']

for name, score in zip(suppliers, final_scores):
    print(f"{name}: {score:.4f}")

best = suppliers[np.argmax(final_scores)]
print(f"🎯 最优选择: {best}")

#准则层扰动
print('扰动后新权重')
r_quality_weight=[]
r_fianl_score=[]
for x in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
    r_weight=disturb_weight_figure(w_criteria_geo,0,x)
    r_quality_weight.append(r_weight[0])
    print(r_weight)
    #扰动后得分
    Rfinal_scores = (r_weight[0] * w_suppliers_quality + 
                r_weight[1] * w_suppliers_cost + 
                r_weight[2] * w_suppliers_efficiency)
    r_fianl_score.append(Rfinal_scores)
r_fianl_score=np.array(r_fianl_score)
print(f"扰动后质量权重{r_quality_weight}")

#三家的总得分
Ar_final_score=r_fianl_score[:,0]
Br_final_score=r_fianl_score[:,1]
Cr_final_score=r_fianl_score[:,2]
#扰动后得分


print(f'扰动后得分{r_fianl_score}')

#质量权重 vs三家最终得分”的折线图
plt.figure(figsize=(8,8))
plt.plot(r_quality_weight,Ar_final_score,color='#0F94FA',linewidth=2,linestyle='-',marker='o',markersize=6,label='A商家')
plt.plot(r_quality_weight,Br_final_score,color="#F40909",linewidth=2,linestyle='-',marker='o',markersize=6,label='B商家')
plt.plot(r_quality_weight,Cr_final_score,color="#00FF22",linewidth=2,linestyle='-',marker='o',markersize=6,label='C商家')
plt.title('质量权重 vs三家最终得分')
plt.xlabel('质量权重')
plt.ylabel('最终得分')
plt.legend()
#plt.grid(True,alpha=0.5)
plt.savefig('灵敏度分析2.pdf')