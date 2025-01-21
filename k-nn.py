import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset from a CSV file
file_path = 'G:\\Mechin learning lab\\iris.csv'
df = pd.read_csv(file_path)

# Assuming the CSV file has the same structure as the Iris dataset
# First 4 columns are features, and the last column is the target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode the string labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Create the k-NN classifier
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plotting the decision boundaries (optional for visualization)
def plot_decision_boundaries(X, y, k):
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = plt.cm.Pastel1
    cmap_bold = plt.cm.Set1

    # Fit the classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    # Plot the decision boundary by assigning a color in the color map
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Reshape Z to match the grid shape
    Z = Z.reshape(xx.shape)

    # Put the result into a color plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"3-Class classification (k = {k})")
    plt.show()

# For demonstration, we will use only the first two features of the dataset to visualize
plot_decision_boundaries(X_train[:, :2], y_train, k)
