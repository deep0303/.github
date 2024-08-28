# model.py
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
df = load_iris()
data = pd.DataFrame(df.data, columns=df.feature_names)

# Select the first feature for X and the last feature for y
x = data.iloc[:, :1]  # sepal length (cm)
y = data.iloc[:, -1]  # petal width (cm)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Standardize the dataset
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

pickle.dump(lr ,open("model.pkl",'wb'))
