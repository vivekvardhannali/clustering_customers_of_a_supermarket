import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# URL of the dataset
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'

# TASK 1: Load the data into a DataFrame. Provide the correct method and arguments.
df = pd.read_excel(data_url)
print(df.isnull().sum())
#there are empty cells in the dataset we have to fill them
# TASK 2: Convert 'InvoiceDate' to datetime format.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# TASK 3: Calculate the total bill for each transaction.
df['Total_Bill'] = df['Quantity'] * df['UnitPrice']

print(df.isnull().sum())
customer_df = df.groupby('CustomerID').agg(
    Total_Bill_Size=('Total_Bill', 'sum'),
    First_Purchase=('InvoiceDate', 'min'),
    Last_Purchase=('InvoiceDate', 'max'),
    Most_Common_Location=('Country', lambda x: x.mode()[0]),
    Top_Item=('StockCode', lambda x: x.value_counts().idxmax())
)
customer_df['Purchase_Interval_Days'] = (customer_df['Last_Purchase'] - customer_df['First_Purchase']).dt.days

from sklearn.impute import SimpleImputer

# Drop date columns
customer_df = customer_df.drop(columns=['First_Purchase', 'Last_Purchase'])

# Handle missing values
# We'll use SimpleImputer for numerical and categorical separately
# Impute numeric columns
num_cols = ['Total_Bill_Size', 'Purchase_Interval_Days']
num_imputer = SimpleImputer(strategy='mean')
customer_df[num_cols] = num_imputer.fit_transform(customer_df[num_cols])

# Impute categorical columns
cat_cols = ['Most_Common_Location', 'Top_Item']
cat_imputer = SimpleImputer(strategy='most_frequent')
customer_df[cat_cols] = cat_imputer.fit_transform(customer_df[cat_cols])

customer_df_encoded = pd.get_dummies(customer_df, columns=cat_cols)

scaler = StandardScaler()
customer_df_encoded[num_cols] = scaler.fit_transform(customer_df_encoded[num_cols])

#Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
customer_df_encoded['Cluster'] = kmeans.fit_predict(customer_df_encoded)

customer_df['Cluster'] = customer_df_encoded['Cluster']
print(customer_df_encoded)
def summarize_cluster_info(clustered_df):
    for i in range(kmeans.n_clusters):
        cluster_data = clustered_df[clustered_df['Cluster'] == i]
        print(f"\nCluster {i} Summary:")

        # Customer count in each cluster
        customer_count = len(cluster_data)
        print(f"Number of Customers in the cluster: {customer_count}")

        # Average spend in each cluster
        avg_spend = cluster_data['Total_Bill_Size'].mean()
        print(f"Average Spend: ${avg_spend:.2f}")

        # Top 3 Locations with counts
        top_locations = cluster_data['Most_Common_Location'].value_counts().head(3)
        print("Top 3 Locations:")
        for location, count in top_locations.items():
            print(f"{location}: {count} customers")

        # Top 3 Items with counts
        top_items = cluster_data['Top_Item'].value_counts().head(3)
        print("Top 3 Item Codes:")
        for item, count in top_items.items():
            print(f"{item}: {count} times purchased")

# Call the function to display the summary
summarize_cluster_info(customer_df)
from sklearn.cluster import DBSCAN
# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)

customer_df=customer_df.drop(columns=[ 'Cluster'])
customer_df_encoded=customer_df_encoded.drop(columns=[ 'Cluster'])
clusters = dbscan.fit_predict(customer_df_encoded)
customer_df['Cluster'] = clusters
# Summary of results
def summarize_clusters2(df):
    grouped = df.groupby('Cluster')
    for key, group in grouped:
        if key == -1:
            continue  # Skip the noise points for detailed summary
        print(f"\nCluster {key} Summary:")
        print(f"Number of Customers: {len(group)}")
        avg_bill = group['Total_Bill_Size'].mean()
        avg_interval = group['Purchase_Interval_Days'].mean()
        print(f"Average Total Bill: {avg_bill:.2f}")
        print(f"Average Purchase Interval: {avg_interval:.2f}")

summarize_clusters2(customer_df)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Create a color map for up to 94 unique cluster labels
colors = plt.cm.tab20(np.linspace(0, 1, 94))  # 94 distinct colors from tab20
cmap = ListedColormap(colors)

plt.figure(figsize=(10, 6))
plt.scatter(
    customer_df_encoded['Total_Bill_Size'], 
    customer_df_encoded['Purchase_Interval_Days'], 
    c=clusters, cmap=cmap, s=5
)

plt.title('DBSCAN Clustering')
plt.xlabel('Scaled Total Bill Size')
plt.ylabel('Scaled Purchase Interval Days')
plt.colorbar(label='Cluster Label')
plt.show()
