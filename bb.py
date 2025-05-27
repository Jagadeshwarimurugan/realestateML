
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

# Load the dataset
df = pd.read_csv('real_estate123.csv')

# Generate a correlation heatmap
def generate_heatmap():
    # Select only the numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate the correlation matrix for numeric columns
    correlation_matrix = numeric_df.corr()

    # Generate the heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')

    # Save the heatmap to a PNG image and encode it to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode('utf-8')
    return chart_url

# Example of model comparison (you can modify this based on your logic)
def compare_models():
    # Here, you'd typically train models and compute the Mean Squared Error (MSE)
    # Example logic for comparing models (dummy values)
    lr_mse = 0.25  # Example MSE for linear regression
    dt_mse = 0.35  # Example MSE for decision tree
    return lr_mse, dt_mse
