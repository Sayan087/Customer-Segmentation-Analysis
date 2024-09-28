import matplotlib
matplotlib.use('Agg')  #

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('Dataset.csv')

# Print the total number of records
print(f'Total records in the dataset: {data.shape[0]}')

# Check for missing values
print('Missing values before filling:')
print(data.isnull().sum())

# Identify numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns

data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
print('Missing values after filling:')
print(data.isnull().sum())

# Count the number of numeric columns
num_numeric_cols = len(numeric_cols)
print(f'Total number of numeric columns: {num_numeric_cols}')
print(f'Total number of plots to be generated: {num_numeric_cols + 1}')  # +1 for the heatmap

# Feature Scaling
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Exploratory Data Analysis (EDA)
# Visualize distributions of key numerical variables
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f'distribution_{col}.png')  # Save the plot as an image file
    plt.close()  # Close the figure to avoid memory issues

# Calculate the correlation matrix
corr_matrix = data[numeric_cols].corr()

# Visualize the correlation matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')  # Save the heatmap as an image file
plt.close()

# Clustering
# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(data[numeric_cols])
kmeans_silhouette = silhouette_score(data[numeric_cols], data['KMeans_Cluster'])
print(f'Silhouette Score for K-Means: {kmeans_silhouette:.2f}')

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
data['DBSCAN_Cluster'] = dbscan.fit_predict(data[numeric_cols])

# Evaluating DBSCAN (Not all clusters may be valid)
if len(set(data['DBSCAN_Cluster'])) > 1:  # Check if more than one cluster is formed
    dbscan_silhouette = silhouette_score(data[numeric_cols], data['DBSCAN_Cluster'])
    print(f'Silhouette Score for DBSCAN: {dbscan_silhouette:.2f}')
else:
    print('DBSCAN did not form any clusters.')

# Check the data types in the DataFrame
print("Data types in the DataFrame:")
print(data.dtypes)

# Cluster Profiling - Ensure only numeric columns are used
numeric_for_profiling = data[numeric_cols].select_dtypes(include=['number'])
kmeans_profile = numeric_for_profiling.groupby(data['KMeans_Cluster']).mean()
print('K-Means Cluster Profiles:')
print(kmeans_profile)

# Propose marketing strategies based on cluster profiles
for cluster_num in kmeans_profile.index:
    print(f"\nMarketing strategies for Cluster {cluster_num}:")
    # Customize your strategies based on cluster profile insights
    print(" - Tailor marketing campaigns based on average transaction patterns.")
    print(" - Offer personalized rewards based on cluster characteristics.")
    print(" - Conduct follow-up surveys to gather feedback.")

# Print the shape of the final DataFrame
print(f'Total records after processing: {data.shape[0]}')
