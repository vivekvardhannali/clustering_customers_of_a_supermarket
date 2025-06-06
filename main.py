import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# URL of the dataset
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'

#Loading the data into a DataFrame.
df = pd.read_excel(data_url)
#checking if there are empty cells in the dataset we have to fill them
print(df.isnull().sum())
# Convert 'InvoiceDate' field to datetime format.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#Calculating the total bill for each transaction.
df['Total_Bill'] = df['Quantity'] * df['UnitPrice']
#grouping all df according to the customers
customer_df = df.groupby('CustomerID').agg(
    Total_Bill_Size=('Total_Bill', 'sum'),
    First_Purchase=('InvoiceDate', 'min'),
    Last_Purchase=('InvoiceDate', 'max'),
    Most_Common_Location=('Country', lambda x: x.mode()[0]),
    Top_Item=('StockCode', lambda x: x.value_counts().idxmax())
)
#adding new coloumn to the dataframe pruchase interval days so that i can pass this as a feature 
#and it is used as argument bcz if they bought only once the value will be zero and if they cameback it will be positive
#it can say about customer loyalty long term repeat customers and if short term impulse buyers and it also tells the 
#span over which buyer did that total bill which helps to evaluate their buying capacity
customer_df['Purchase_Interval_Days'] = (customer_df['Last_Purchase'] - customer_df['First_Purchase']).dt.days

from sklearn.impute import SimpleImputer

# Drop date columns as they are no longer used
customer_df = customer_df.drop(columns=['First_Purchase', 'Last_Purchase'])

# Handling the  missing values
# We'll use SimpleImputer for numerical and categorical separately
# Imputing numeric columns
num_cols = ['Total_Bill_Size', 'Purchase_Interval_Days']
num_imputer = SimpleImputer(strategy='mean')
customer_df[num_cols] = num_imputer.fit_transform(customer_df[num_cols])

# Imputing categorical columns
cat_cols = ['Most_Common_Location', 'Top_Item']
cat_imputer = SimpleImputer(strategy='most_frequent')
customer_df[cat_cols] = cat_imputer.fit_transform(customer_df[cat_cols])
#encoding the categorical coloumns 
customer_df_encoded = pd.get_dummies(customer_df, columns=cat_cols)
#standardizing customer_df_encoded
scaler = StandardScaler()
customer_df_encoded[num_cols] = scaler.fit_transform(customer_df_encoded[num_cols])

#Apply K-Means
cluster_count=5
kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
#adding the cluster indices coloumn to the data
customer_df_encoded['Cluster'] = kmeans.fit_predict(customer_df_encoded)

customer_df['Cluster'] = customer_df_encoded['Cluster']
print(customer_df_encoded)
#printing the clusters
def summarize_cluster_info(clustered_df):
    for i in range(kmeans.n_clusters):
        cluster_data = clustered_df[clustered_df['Cluster'] == i]
        print(f"\nCluster {i} Summary:")

        customer_count = len(cluster_data)
        print(f"Number of Customers in the cluster: {customer_count}")

        avg_spend = cluster_data['Total_Bill_Size'].mean()
        print(f"Average Spend: ${avg_spend:.2f}")

        top_locations = cluster_data['Most_Common_Location'].value_counts().head(3)
        print("Top 3 Locations:")
        for location, count in top_locations.items():
            print(f"{location}: {count} customers")

        top_items = cluster_data['Top_Item'].value_counts().head(3)
        print("Top 3 Item Codes:")
        for item, count in top_items.items():
            print(f"{item}: {count} times purchased")

# Call the function to display the summary
summarize_cluster_info(customer_df)
#clustering using dbscan method
from sklearn.cluster import DBSCAN
# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)

customer_df=customer_df.drop(columns=[ 'Cluster'])
customer_df_encoded=customer_df_encoded.drop(columns=[ 'Cluster'])
clusters = dbscan.fit_predict(customer_df_encoded)
customer_df['Cluster'] = clusters
# Summary of results
print("USING DBSCAN TO FORM THE CLUSTERS\n")
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
