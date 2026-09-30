from scipy.optimize import linprog

# 目标函数系数：求最大值，linprog 默认求最小值，所以取负号
c = [-40, -30]

# 不等式约束 A_ub x <= b_ub
A_ub = [
    [2, 1],   # 2x1 + x2 <= 100
    [1, 1],   # x1 + x2 <= 80
    [1, 0],   # x1 <= 40
]
b_ub = [100, 80, 40]

# 变量取值范围：x1 >= 0, x2 >= 0
x_bounds = (0, None)
y_bounds = (0, None)

result = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[x_bounds, y_bounds],
    method="highs"
)

print(result)
print("是否成功：", result.success)
print("甲产量 x1 =", result.x[0])
print("乙产量 x2 =", result.x[1])
print("最大利润 =", -result.fun)