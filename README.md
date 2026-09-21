# RameshDemo

# Optimized final
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score
 
# 1. Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_ids = test_df['id']
 
X = train_df.drop(columns=['y'])
y = train_df['y'].map({'yes': 1, 'no': 0})
X_test_data = test_df.drop(columns=['id'])
 
# One-Hot Encoding
combined_features = pd.concat([X, X_test_data], keys=['train', 'test'])
# combined_features = combined_features.drop(columns=['duration'])
combined_encoded = pd.get_dummies(combined_features, drop_first=True)
 
X_encoded = combined_encoded.xs('train')
X_test_final = combined_encoded.xs('test')
 
# Train test data Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
 
# Scaling numerical entries
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test_final = scaler.transform(X_test_final)
 
print("Evaluating best Models.......")
 
# 1. KNN
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)
print(f"KNN Macro F1: {f1_score(y_test, knn_preds, average='macro'):.4f}")
 
# 2. Decision Tree
dtree = DecisionTreeClassifier(class_weight='balanced')
dtree.fit(X_train, y_train)
dt_preds = dtree.predict(X_test)
print(f"Decision Tree Macro F1: {f1_score(y_test, dt_preds, average='macro'):.4f}")
 
# 3. Logistic Regression
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)
print(f"Logistic Regression Macro F1: {f1_score(y_test, log_preds, average='macro'):.4f}")
 
print("--- KNN Report ---")
print(classification_report(y_test, knn_preds, target_names=['no', 'yes']))
 
print("--- Decision Tree Report ---")
print(classification_report(y_test, dt_preds, target_names=['no', 'yes']))
 
print("--- Logistic Regression Report ---")
print(classification_report(y_test, log_preds, target_names=['no', 'yes']))
 
# Generate final predictions on csv file
final_predictions_numeric = log_reg.predict(X_test_final)
 
# Convert numeric predictions (1/0) back to original format ('yes'/'no')
final_predictions_labels = np.where(final_predictions_numeric == 1, 'yes', 'no')
 
output_df = pd.DataFrame({
    'id': test_ids,
    'y': final_predictions_labels
})
output_df.to_csv('optimized_predictions.csv', index=False)
print("\nHighly optimized predictions successfully saved to 'optimized_predictions.csv'!")
 
 
 
here is the coe used for customer campaign model
 
