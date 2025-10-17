import pandas as pd
import numpy as np
df = pd.read_csv("data/windows_aligned.csv")
# 若文件里已有 center_dev 列就直接用；若没有，用 a/v 中心差除以窗口长自己算
if "center_dev" in df.columns:
    x = df.loc[df["keep"]==1, "center_dev"].values
    print("p50=%.3f p90=%.3f p95=%.3f max=%.3f" % (np.percentile(x,50), np.percentile(x,90), np.percentile(x,95), x.max()))
