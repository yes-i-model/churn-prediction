import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Load the dataset (replace 'dataset.csv' with your dataset file)
data = pd.read_csv('dataset.csv')

# Data preprocessing
# Drop irrelevant columns and handle missing values
data = data.drop(['customer_id', 'other_irrelevant_columns'], axis=1)
data = data.dropna()

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split the dataset into features (X) and target (y)
X = data.drop('churn', axis=1)
y = data['churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

customer_ids = data['customer_id']  # Assuming 'customer_id' is the column containing Customer IDs
X_new = scaler.transform(X_new)  # Assuming X_new is the data for new customers to predict
churn_probabilities = model.predict_proba(X_new)[:, 1]

# Create a dataframe with Customer IDs and their churn probabilities
output_df = pd.DataFrame({'Customer ID': customer_ids, 'Predicted Probability of Churn': churn_probabilities})
print(output_df)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
