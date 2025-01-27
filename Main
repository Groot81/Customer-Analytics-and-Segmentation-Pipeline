# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics  # Import for Davies-Bouldin index

# Task 1: Load and Clean Data
def load_data():
    customers = pd.read_csv("Customers.csv")
    products = pd.read_csv("Products.csv")
    transactions = pd.read_csv("Transactions.csv")
    return customers, products, transactions

def clean_data(customers, products, transactions):
    # Check for missing values
    print("Missing values in Customers:", customers.isnull().sum())
    print("Missing values in Products:", products.isnull().sum())
    print("Missing values in Transactions:", transactions.isnull().sum())

    # Drop duplicates
    customers.drop_duplicates(inplace=True)
    products.drop_duplicates(inplace=True)
    transactions.drop_duplicates(inplace=True)

    return customers, products, transactions

# Task 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis(customers, products, transactions):
    # Example plots
    # Customer region distribution
    sns.countplot(y='Region', data=customers)
    plt.title("Customer Region Distribution")
    plt.show()

    # Product category distribution
    sns.countplot(y='Category', data=products)
    plt.title("Product Category Distribution")
    plt.show()

    # Sales over time
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    transactions.groupby(transactions['TransactionDate'].dt.to_period('M')).sum()['TotalValue'].plot()
    plt.title("Monthly Sales")
    plt.show()

# Task 3: Lookalike Model
def lookalike_model(customers, transactions):
    # Merge data
    customer_transactions = transactions.merge(customers, on='CustomerID')

    # Select relevant columns for cosine similarity (you may need to adjust these columns)
    # Example: Using numeric features for similarity, adjust based on your dataset
    customer_features = customer_transactions[['Feature1', 'Feature2']]  # Replace with actual feature names

    # Calculate similarities (example using selected features)
    similarity_matrix = cosine_similarity(customer_features)

    # Get top 3 lookalikes for the first 20 customers
    lookalike_dict = {}
    for i in range(20):
        similar_indices = similarity_matrix[i].argsort()[-4:-1][::-1]  # Top 3 excluding self
        lookalike_dict[customers['CustomerID'].iloc[i]] = [(customers['CustomerID'].iloc[j], similarity_matrix[i, j]) for j in similar_indices]

    # Save as CSV
    lookalike_df = pd.DataFrame([(key, value) for key, values in lookalike_dict.items() for value in values],
                                columns=['CustomerID', 'Lookalike'])
    lookalike_df.to_csv("Lookalike.csv", index=False)

# Task 4: Clustering
def clustering(customers, transactions):
    # Combine customer profile and transaction history
    customer_data = transactions.groupby('CustomerID').sum().merge(customers, on='CustomerID')

    # Normalize data (excluding CustomerID)
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data.iloc[:, 1:])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

    # Evaluate clusters
    print("Cluster Centers:", kmeans.cluster_centers_)
    print("Davies-Bouldin Index:", metrics.davies_bouldin_score(customer_data_scaled, customer_data['Cluster']))

    # Visualize the clusters (if you have more than 2 features, consider using PCA for 2D visualization)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=customer_data_scaled[:, 0], y=customer_data_scaled[:, 1], hue=customer_data['Cluster'])
    plt.title("Customer Clusters")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and clean data
    customers, products, transactions = load_data()
    customers, products, transactions = clean_data(customers, products, transactions)

    # Task 1: Perform EDA
    exploratory_data_analysis(customers, products, transactions)

    # Task 2: Lookalike Model
    lookalike_model(customers, transactions)

    # Task 3: Customer Segmentation
    clustering(customers, transactions)
