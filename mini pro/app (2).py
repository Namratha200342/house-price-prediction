from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Initialize Flask app
app = Flask(__name__)

# Example dataset (replace with your own dataset)
data = {
    'Area': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
    'Age': [10, 12, 15, 20, 8, 15, 7, 5, 10, 12],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

# Converting dataset to DataFrame
df = pd.DataFrame(data)

# Feature selection (Independent variables)
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation and training
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Ensure the static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Save the plot as an image
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red')  # line for perfect prediction
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plot_path = "static/plot.png"
plt.savefig(plot_path)
plt.close()

@app.route('/')
def home():
    # Pass data to the HTML template
    return render_template('index.html', 
                           predicted_prices=y_pred, 
                           actual_prices=list(y_test), 
                           rmse=rmse, 
                           plot_path=plot_path)

if __name__ == "__main__":
    app.run(debug=True)
