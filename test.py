import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
s = np.linspace(1, 0, 101)
x = np.random.randn()

xs = [x]       # x after each step
incs = []      # ds * v of each step

for i in range(len(s) - 1):
    ds = s[i+1] - s[i]
    v = np.random.uniform()
    inc = ds * v
    x = x + inc
    incs.append(inc)
    xs.append(x)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# 1) per-step increment (ds * v)
axes[0].bar(range(len(incs)), incs, color="tab:red")
axes[0].set_title("per-step increment: ds * v  (always negative since ds < 0)")
axes[0].axhline(0, color="k", lw=0.5)

# 2) accumulated x
axes[1].bar(range(len(xs)), xs, color="tab:blue")
axes[1].set_title("x after each step (accumulated)")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_xlabel("step")

plt.tight_layout()
plt.savefig("euler_bar.png", dpi=120)
print(xs[0], xs[-1], sum(incs))