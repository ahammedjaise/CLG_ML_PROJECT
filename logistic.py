import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score
import seaborn as sns

data = pd.read_csv(r'C:\Users\HP\Downloads\Iris.csv')
x= data.drop('Species', axis=1)
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)
model = LogisticRegression(max_iter=100)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred, average='macro'))

cm = confusion_matrix(y_test, y_pred)
labels = model.classes_

plt.figure(figsize=(6,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()  
