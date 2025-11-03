import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report,precision_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

data=load_iris()
x=data.data
y=data.target
target_names= data.target_names

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print('accuracy:\n',accuracy_score(y_test,y_pred))
print('\nprecision:\n',precision_score(y_test,y_pred,average='macro'))
print('\nclass_report:\n',classification_report(y_test,y_pred,target_names=target_names))
print('\ncm:\n',confusion_matrix(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
labels=model.classes_

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels)
plt.title('CM')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
