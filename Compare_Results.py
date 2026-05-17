import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# ----------------------
# 1. Data Input (replace with your data)
# ----------------------
# a: 30 results from Method A (floating-point values)
# b: 30 paired results from Method B (floating-point values)
# Ensure the i-th element of a corresponds to the i-th element of b (paired structure)
data = pd.read_csv('.\Data/1102/D3QNtest_Error.csv')
a =  np.asarray(data['Value'])
data = pd.read_csv('.\Data/1102/SL_Error.csv')
b = np.asarray(data['Value'])


# ----------------------
# 2. Visualization: Paired Data Comparison
# ----------------------
plt.figure(figsize=(10, 6))

# Boxplot for overall distribution
plt.subplot(1, 2, 1)
plt.boxplot([a, b], labels=['Method A', 'Method B'], patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Distribution of Results')
plt.ylabel('Value (Smaller = Better)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Scatter plot for paired relationships
plt.subplot(1, 2, 2)
plt.scatter(range(1, 31), a, label='Method A', color='blue', marker='o')
plt.scatter(range(1, 31), b, label='Method B', color='orange', marker='x')
plt.title('Paired Results (by Experiment)')
plt.xlabel('Experiment Index')
plt.ylabel('Value')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# ----------------------
# 3. Paired Sample Hypothesis Testing
# ----------------------
alpha = 0.05  # Significance level
d = a - b  # Calculate differences (A - B) for paired analysis

# Basic statistics
print("\n===== Paired Sample Statistics =====")
print(f"Mean of Method A: {np.mean(a):.4f} (SD: {np.std(a, ddof=1):.4f})")
print(f"Mean of Method B: {np.mean(b):.4f} (SD: {np.std(b, ddof=1):.4f})")
print(f"Mean difference (A - B): {np.mean(d):.4f} (SD: {np.std(d, ddof=1):.4f})")
print(f"Proportion of experiments where A < B: {np.mean(a < b):.2%}")


# Normality test for differences (critical for choosing test type)
shapiro_stat, shapiro_p = stats.shapiro(d)
print(f"\n===== Normality Test (Shapiro-Wilk) on Differences =====")
print(f"Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
print("Interpretation: If p > 0.05, differences are approximately normally distributed.")


# Select appropriate test based on normality
if shapiro_p > alpha:
    # Normal distribution: Use paired t-test
    test_name = "Paired t-test"
    _, p_value = stats.ttest_rel(a, b, alternative='less')  # H1: A < B
else:
    # Non-normal distribution: Use Wilcoxon signed-rank test
    test_name = "Wilcoxon signed-rank test (non-parametric)"
    _, p_value = stats.wilcoxon(a, b, alternative='less')  # H1: A < B


# ----------------------
# 4. Conclusion
# ----------------------
print(f"\n===== Test Result =====")
print(f"Test used: {test_name}")
print(f"p-value: {p_value:.6f}")
print(f"Significance level: α = {alpha}")

if p_value < alpha:
    print(f"Conclusion: Reject the null hypothesis. Method A is significantly better than Method B (A's results are statistically smaller) at the α={alpha} level.")
else:
    print(f"Conclusion: Fail to reject the null hypothesis. There is insufficient evidence to conclude Method A is better than Method B at the α={alpha} level.")