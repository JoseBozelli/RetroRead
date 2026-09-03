"""
Calls make_gaussian_heatmap directly with typical values to see exactly
where the overflow warning originates.

Run from the repo root with:
    uv run python scripts/exploratory/debug_gaussian_function.py
"""

import numpy as np

from retroread.heatmap_dataset import make_gaussian_heatmap

np.seterr(all="raise")  # turn warnings into actual errors, so we get a traceback

try:
    result = make_gaussian_heatmap(cx=15.0, cy=10.0, size=28, sigma=1.5)
    print(f"Success. dtype={result.dtype}  min={result.min()}  max={result.max()}")
except Exception as e:
    import traceback
    traceback.print_exc()

# Also check intermediate dtype without the final cast
y, x = np.mgrid[0:28, 0:28]
print(f"\nmgrid dtype: {x.dtype}")
diff = x - 15.0
print(f"(x - cx) dtype: {diff.dtype}")
squared = diff ** 2
print(f"squared dtype: {squared.dtype}  max value: {squared.max()}")