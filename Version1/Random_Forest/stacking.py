from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from data_util import get_clf_data
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"   

x_train,x_test,y_train,y_test=get_clf_data()

estimators=[
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('knn', KNeighborsClassifier())
]

stack=StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)
stack.fit(x_train,y_train)
print("Stacking:", accuracy_score(y_test, stack.predict(x_test)))

stack_plus = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=True
)
stack_plus.fit(x_train, y_train)
print("Stacking+原始特征:", accuracy_score(y_test, stack_plus.predict(x_test)))