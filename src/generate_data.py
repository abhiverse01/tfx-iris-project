# src/generate_data.py
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Save dataset as CSV
df.to_csv('data/iris_data.csv', index=False)
