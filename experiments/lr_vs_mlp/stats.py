import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


stats = pd.read_csv("chunk_stats_091724.csv") 
print(stats)

# print(stats.drop("_label", axis=1).std())

x = np.arange(10)
for _label in stats["_label"].unique():
  tmp = stats[stats["_label"] == _label].drop("_label", axis=1)
  avg, std = tmp.mean(), tmp.std()
  print(_label, avg.tolist(), std.tolist())

  plt.errorbar(x, avg.tolist(), std.tolist(), marker="^", label=_label)


plt.xticks(x)
plt.xlabel("Harmful -> Helpful")
plt.ylabel("Target Proportions")
plt.ylim(0,1)
plt.legend(frameon=False)
plt.show()