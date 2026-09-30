import pulp

prob = pulp.LpProblem("MaxProfit", pulp.LpMaximize)

x1 = pulp.LpVariable("x1", lowBound=0, cat="Integer")
x2 = pulp.LpVariable("x2", lowBound=0, cat="Integer")

prob += 40 * x1 + 30 * x2          # 目标函数
prob += 2 * x1 + x2 <= 100         # 机器时间
prob += x1 + x2 <= 80              # 原料
prob += x1 <= 40                   # 甲上限

prob.solve()

print("状态：", pulp.LpStatus[prob.status])
print("甲产量 =", x1.varValue)
print("乙产量 =", x2.varValue)
print("最大利润 =", pulp.value(prob.objective))