import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.datasets import load_iris

data=load_iris()
#data=data.dropna()
#data=data.drop_duplicates()

x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='macro')
cm=confusion_matrix(y_test,y_pred)
correct=y_pred==y_test
wrong=y_pred!=y_test

print(f"model accuracy: {accuracy:.2f}")
print(f"precision score: {precision:.2f}")
print("Confusion Matrix:")
print(cm)
print("classification report=", classification_report(y_test,y_pred))
print("correct predictions:", np.sum(correct))
print("wrong predictions:", np.sum(wrong))
print("\ncorrectprediction:")
print(np.column_stack((y_test[correct], y_pred[correct])))
print("\nwrongprediction:")
print(np.column_stack((y_test[wrong], y_pred[wrong])))

label=sorted(np.unique(y))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)            
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()