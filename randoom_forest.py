# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the wine quality dataset
file_path = r"G:\Mechin learning lab\winequalityN.csv"
df = pd.read_csv(file_path)

# Inspect the dataset (optional)
print(df.head())

# Convert categorical variable into dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Assuming the last column is the target and the rest are features
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fitting the model
rf_classifier.fit(X_train, y_train)

# Making predictions
y_pred = rf_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
