
import numpy as np
import matplotlib.pyplot as plt

methods = [
    "Verified GA vs Fixed",
    "Verified GA vs Ordinary Random",
    "Verified GA vs Ordinary GA",
    "Verified GA vs Verified Random",
    "Adjacent GA vs Verified GA",
]

improvement = np.array([2.14, 1.13, 0.73, -0.41, -0.51])
lower = np.array([-2.50, -0.86, -0.69, -1.37, -1.03])
upper = np.array([6.17, 3.18, 2.21, 0.49, -0.004])

y = np.arange(len(methods))
errors = np.vstack([improvement - lower, upper - improvement])

fig, ax = plt.subplots(figsize=(9, 4.2))

ax.errorbar(
    improvement,
    y,
    xerr=errors,
    fmt="o",
    capsize=5,
    markersize=6,
    linewidth=1.8,
)

ax.axvline(0, linestyle="--", linewidth=1, alpha=0.7)

ax.set_yticks(y)
ax.set_yticklabels(methods)
ax.invert_yaxis()

ax.set_xlabel("Relative improvement (%)")
ax.set_title("Paired Comparisons with 95% Confidence Intervals")
ax.set_xlim(-3.0, 7.0)
ax.grid(axis="x", linestyle=":", alpha=0.35)

plt.tight_layout()
plt.savefig("paired_comparisons.pdf", bbox_inches="tight")
plt.savefig("paired_comparisons.png", dpi=300, bbox_inches="tight")
plt.show()
