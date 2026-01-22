from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Creating the boxplot
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))
df.boxplot(rot = 45)
plt.tight_layout()

# Saving the boxplot
plt.savefig("figs/california_housing_boxplot.png")