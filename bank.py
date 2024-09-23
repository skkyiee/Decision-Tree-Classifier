import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

file_path = "bank-additional.csv"
data = pd.read_csv(file_path, delimiter=';')

# Display the first few rows
print(data.head())

# Encoding categorical variables
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == type(object):
        data[column] = le.fit_transform(data[column])

# Split the data into features (X) and target (y)
X = data.drop('y', axis=1)  # Features
y = data['y']  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])

# Print the results in the desired format
print("\n")
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print("Predicted")
print(f"{'':<10} {'0':<10} {'1':<10}")
print(f"{'Actual 0':<10} {conf_matrix[0][0]:<10} {conf_matrix[0][1]:<10}")
print(f"{'Actual 1':<10} {conf_matrix[1][0]:<10} {conf_matrix[1][1]:<10}")
print("\n")
print(clf.criterion)

print("Classification Report:")
print(f"Class 0 (No):\nPrecision: {class_report.splitlines()[2].split()[1]}\nRecall: {class_report.splitlines()[2].split()[2]}\nF1-Score: {class_report.splitlines()[2].split()[3]}")
print(f"\nClass 1 (Yes):\nPrecision: {class_report.splitlines()[3].split()[1]}\nRecall: {class_report.splitlines()[3].split()[2]}\nF1-Score: {class_report.splitlines()[3].split()[3]}")
print(f"\nOverall Accuracy: {accuracy:.2f}")

print(f"\nMacro Average:\nPrecision: {class_report.splitlines()[-2].split()[2]}\nRecall: {class_report.splitlines()[-2].split()[3]}\nF1-Score: {class_report.splitlines()[-2].split()[4]}")
print(f"\nWeighted Average:\nPrecision: {class_report.splitlines()[-1].split()[2]}\nRecall: {class_report.splitlines()[-1].split()[3]}\nF1-Score: {class_report.splitlines()[-1].split()[4]}")

# Plot the decision tree;
plt.figure(figsize=(20, 10))  
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.savefig('decision_tree.png', dpi=1200)  
plt.show()
