import numpy as np
from scipy import stats

N = 10
a = np.random.randn(N) + 2      # mean 2, variance 1
b = np.random.randn(N)          # mean 0, variance 1

var_a = a.var(ddof=1)   # unbiased n-1
var_b = b.var(ddof=1)

s = np.sqrt((var_a + var_b) / 2)
t = (a.mean() - b.mean()) / (s * np.sqrt(2.0/N))    # t-statistic
df = (2 * N) - 2    # degree of freedom

p = 1 - stats.t.cdf(np.abs(t), df=df)   # one-sided p-value
p2 = p * 2                              # two sided p-value

print(t)
print(p, p2)
