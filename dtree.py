import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred= model.predict(x_test)

print('accurac:',accuracy_score(y_test,y_pred))
print('precision:',precision_score(y_test,y_pred))
print('\nclassification report:\n',classification_report(y_test,y_pred))

plt.figure(figsize=(10,20))
plot_tree(model,
    feature_names=data.feature_names.tolist(),
    class_names=data.target_names.tolist(),
    filled=True,rounded=True,fontsize=10)
plt.title('dtree')
plt.tight_layout()
plt.show()
