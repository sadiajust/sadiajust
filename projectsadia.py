import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define the dataset manually
data = {
    'Age': [25, 32, 47, 40, 20, 35, 60, 30, 28, 50],
    'EstimatedSalary': [35000, 57000, 90000, 76000, 20000, 69000, 150000, 50000, 40000, 100000],
    'Purchased': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
}

# Create a DataFrame
dataset = pd.DataFrame(data)
print("The imported data is:\n", dataset.head())

# Introduce some missing values
dataset.loc[0:2, 'Age'] = np.nan

# Show the dataset with inserted anomalies
print("Dataset with inserted blank data:\n", dataset.head())

# Select features and target
features = dataset.iloc[:, [0, 1]].values
target = dataset.iloc[:, 2].values

# Handle missing data using SimpleImputer
imputer = SimpleImputer(strategy='mean')
features[:, 0:1] = imputer.fit_transform(features[:, 0:1])
print("Processed features after handling missing data:\n", features[:5])

# Split the dataset into training and testing sets with stratification
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=0, stratify=target
)

# Apply MinMax scaling
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Show scaled training and testing datasets
print("Scaled training data:\n", features_train[:5])
print("Scaled testing data:\n", features_test[:5])

# Train SVM model with linear kernel
svm_classifier = SVC(kernel='linear', random_state=0)
svm_classifier.fit(features_train, target_train)

# Make predictions
svm_predictions = svm_classifier.predict(features_test)

# Evaluate model performance
svm_confusion_matrix = confusion_matrix(target_test, svm_predictions)
svm_accuracy = accuracy_score(target_test, svm_predictions)

print("SVM Confusion Matrix:")
print(svm_confusion_matrix)
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print("SVM Classification Report:")
print(classification_report(target_test, svm_predictions, zero_division=1))

# Train Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
rf_classifier.fit(features_train, target_train)

# Make predictions
rf_predictions = rf_classifier.predict(features_test)

# Evaluate model performance
rf_confusion_matrix = confusion_matrix(target_test, rf_predictions)
rf_accuracy = accuracy_score(target_test, rf_predictions)

print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print("Random Forest Classification Report:")
print(classification_report(target_test, rf_predictions, zero_division=1))

# Visualize SVM decision boundary
x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                      np.arange(y_min, y_max, 0.01))
zz = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contour(xx, yy, zz, colors='green', linewidths=0.5)
plt.scatter(features_train[:, 0], features_train[:, 1], c=target_train, cmap='plasma', edgecolor='black', s=40)
plt.title('Enhanced SVM Decision Boundary Visualization Without Background')
plt.xlabel('Feature 1 (Scaled Age)')
plt.ylabel('Feature 2 (Scaled Estimated Salary)')
plt.show()
