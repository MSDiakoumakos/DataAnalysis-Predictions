import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    # Remove specified features
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])
    
    # Convert boolean values to numeric (assuming 'Revenue' is the boolean column)
    df['Revenue'] = df['Revenue'].astype(int)
    
    # Apply one-hot encoding to categorical variables
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
    
    # Separate the target variable
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Load the dataset
df = pd.read_csv('project2_dataset.csv')

# Prepare the data with a 70%-30% train-test split and a random seed of 42
X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.7, shuffle=True, random_state=42)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data only and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Logistic Regression model with no penalty and increased max_iter
log_reg = LogisticRegression(penalty='none', max_iter=1000)

# Train the model on the scaled training data
log_reg.fit(X_train_scaled, y_train)

# Make predictions on the training and testing sets
y_train_pred = log_reg.predict(X_train_scaled)
y_test_pred = log_reg.predict(X_test_scaled)

# Calculate accuracy on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Accuracy on the training set: {train_accuracy:.2f}")

# Calculate accuracy on the testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy on the testing set: {test_accuracy:.2f}")

# Generate and print the confusion matrix for the testing set
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for the testing set:")
print(conf_matrix)