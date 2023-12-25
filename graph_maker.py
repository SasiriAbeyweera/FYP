import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read experimental data from Excel
excel_file_path = r"C:\Users\sasir\Desktop\Distancell.xlsx"
df = pd.read_excel(excel_file_path)

# Assuming 'X' is your input feature and 'y' is your real output
X_label = 'Real'
y_label = 'Sigma'
X = df[X_label].values
y = df[y_label].values

# Plot the results
plt.scatter(X, y, color='black', label=y_label)
# plt.plot(X, y, color='blue', linewidth=1, label='Polynomial Fit')
plt.xlabel(X_label)
plt.ylabel(y_label)
plt.title(f'Graph {X_label} vs {y_label}')
plt.legend()
plt.show()
