'''权重'''
import numpy as np
def ahp_geometric_method(matrix):
    n=matrix.shape[0]
    #方根法计算每行几何平均值
    row_geometric_mean=np.prod(matrix,axis=1)**(1/n)
    #归一化
    weights=row_geometric_mean/np.sum(row_geometric_mean)
    return weights

def ahp_sum_product_method(matrix):
    #和积法计算
    n=matrix.shape[0]
    #归一化
    col_sum=np.sum(matrix,axis=0)
    normalized_matrix=matrix/col_sum
    #按行求和
    row_sum=np.sum(normalized_matrix,axis=1)
    #再次归一化
    weights=row_sum/np.sum(row_sum)
    return weights

'''一致性检验'''
def consistency_check(matrix,weights):
    #多少行
    n=matrix.shape[0]
    #计算最大特征值 lambda_max
    # 对于方根法，lambda_max ≈ sum((A*w)_i / w_i) / n
    Aw=np.dot(matrix,weights) #二维矩阵乘法
    lambda_max=np.mean(Aw/weights)
    CI=(lambda_max-n)/(n-1)
    #平均随机一致性指标RI(1-10阶)
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,  6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI=RI_dict.get(n,1.49) # # 若n>10，用1.49近似,).从字典中查找键 n 对应的值；如果找不到，就返回默认值 1.49。
    CR=CI/RI if RI!=0 else 0
    return CR


    

