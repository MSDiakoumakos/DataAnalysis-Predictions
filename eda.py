import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('project2_dataset.csv')

# Number of records in the dataset
num_records = len(data)
print(f"Number of records: {num_records}")

# Percentage of sessions resulting in a purchase
purchase_rate = data['Revenue'].mean() * 100
print(f"Percentage of users who made a purchase: {purchase_rate:.2f}%")

# Accuracy of a model always predicting 'no purchase'
accuracy_no_purchase = (1 - data['Revenue'].mean()) * 100
print(f"Accuracy of always predicting no purchase: {accuracy_no_purchase:.2f}%")


#Visualize the data (It doesnt work :/)
# # Distribution of a feature
# sns.histplot(data['SomeFeature'])
# plt.title('Distribution of SomeFeature')
# plt.show()

# # Relationship with the Revenue
# sns.countplot(x='SomeFeature', hue='Revenue', data=data)
# plt.title('Purchase by SomeFeature')
# plt.show()
