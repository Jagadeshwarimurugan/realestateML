import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import io
import base64

# Load dataset
df = pd.read_csv('real_estate_tamilnadu.csv')

# Function to generate a heatmap of numeric feature correlations
def generate_heatmap():
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')

    # Save heatmap to buffer and encode
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

# Function to compare model performance: Linear, Random Forest, XGBoost
def compare_models():
    # Select features (excluding 'Location')
    X = df[['Square_Feet', 'Bedrooms', 'Age_of_Property']]
    y = df['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_preds)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_preds)

    # XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_preds)

    return {
        'Linear Regression MSE': lr_mse,
        'Random Forest MSE': rf_mse,
        'XGBoost MSE': xgb_mse
    }

# Main
if __name__ == '__main__':
    heatmap_base64 = generate_heatmap()  # Optional
    model_results = compare_models()

    print("📊 Model Performance (Mean Squared Error):")
    for model, mse in model_results.items():
        print(f"{model}: {mse:,.2f}")
