import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'C:\Users\HP\Downloads\Heart.csv')
df = df.dropna()
df['AHD'] = df['AHD'].map({'Yes': 1, 'No': 0})

y = df['AHD']
x = df.drop(columns=['AHD'])

x = pd.get_dummies(x)
print('y value counts before split:')
print(y.value_counts())
# Split data with stratification to preserve class balance
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
print('y_train value counts after split:')
print(y_train.value_counts())
# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = SVC(kernel='linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)   
precision = precision_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
