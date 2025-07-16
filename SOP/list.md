```python
import numpy as np

def extract_segment(arr, lo=500, hi=3000, min_len=3):
    """
    從一維 array 中抓出值落在 [lo, hi] 的連續區間。
    回傳 (seg_values, start_idx, end_idx)。抓不到就回空陣列。
    """
    arr = np.asarray(arr, dtype=float)
    mask = (arr >= lo) & (arr <= hi)
    if not mask.any():
        return np.array([]), None, None

    # 找連續 True 的段落
    idx = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)

    # 取最長（或你想要的條件）
    target = max(groups, key=len)
    return arr[target]



lists = [ 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1749, 1749, 1749, 1749, 1749, 1749, 1749, 1749, 1749, 1749, 1749, 1749, 1749, 1749, 1749,1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4, 1, -1.4 ]

print(extract_segment(lists))
```
