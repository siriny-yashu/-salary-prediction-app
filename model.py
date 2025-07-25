import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.DataFrame({
    'Experience': [1, 2, 3, 4, 5],
    'EducationLevel': [1, 2, 3, 2, 1],
    'Salary': [30000, 40000, 60000, 50000, 45000]
})

X = df[['Experience', 'EducationLevel']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
