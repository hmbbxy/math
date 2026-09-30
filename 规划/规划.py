from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import numpy as np
import matplotlib as plt

# LP：linprog 只做 min，所以 max 要先取负
res = linprog(c=[-3, -2], A_ub=np.array([[2,1],[1,3]]), b_ub=[10, 12],
              bounds=[(0, None)]*2, method="highs")
Z = -res.fun                    # 16.4
res.ineqlin.marginals           # 对偶值，取负号就是影子价格
res.ineqlin.residual            # 松弛量 slack

# IP：同一模型加 integrality
res2 = milp(c=[-3, -2], constraints=LinearConstraint([[2,1],[1,3]], -np.inf, [10,12]),
            integrality=[1, 1], bounds=Bounds([0,0], [np.inf]*2))
# Z = 16, 解 (4, 2)
print(res2)

