import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset from your file path
file_path = r'G:\Mechin learning lab\covid_flu.xlsx'
data = pd.read_excel(file_path)

# Display the original data
print("Original Data:")
print(data)

# Convert 'yes'/'no' to 1/0
data['covid'] = data['covid'].map({'yes': 1, 'no': 0})
data['flu'] = data['flu'].map({'yes': 1, 'no': 0})
data['fever'] = data['fever'].map({'yes': 1, 'no': 0})

# Display the converted data
print("\nConverted Data:")
print(data)

# Define the feature (fever) and the target (covid, flu)
X = data[['fever']]  # Feature
y_covid = data['covid']  # Target for covid
y_flu = data['flu']  # Target for flu

# Splitting the data into train and test sets (80% train, 20% test)
X_train, X_test, y_covid_train, y_covid_test = train_test_split(X, y_covid, test_size=0.2, random_state=42)
_, _, y_flu_train, y_flu_test = train_test_split(X, y_flu, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model_covid = GaussianNB()
model_flu = GaussianNB()

# Train the model for COVID
model_covid.fit(X_train, y_covid_train)

# Train the model for Flu
model_flu.fit(X_train, y_flu_train)

# Make predictions
covid_predictions = model_covid.predict(X_test)
flu_predictions = model_flu.predict(X_test)

# Get prediction probabilities
covid_probabilities = model_covid.predict_proba(X_test)
flu_probabilities = model_flu.predict_proba(X_test)

# Find the maximum predicted probability for each class (COVID and Flu)
max_covid_prob = covid_probabilities.max(axis=1)
max_flu_prob = flu_probabilities.max(axis=1)

print("\nMax probabilities for COVID:", max_covid_prob)
print("Max probabilities for Flu:", max_flu_prob)

# Optionally check accuracy
covid_accuracy = accuracy_score(y_covid_test, covid_predictions)
flu_accuracy = accuracy_score(y_flu_test, flu_predictions)

print("\nAccuracy for COVID:", covid_accuracy)
print("Accuracy for Flu:", flu_accuracy)

