from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 写你机器的真实核心数


x,y=make_classification(n_samples=1000,n_features=16,n_classes=2,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

bagging=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bagging.fit(x_train,y_train)
print("acc=",bagging.score(x_test,y_test))

rf=RandomForestClassifier(n_estimators=100,
                          random_state=42)
rf.fit(x_train,y_train)
print("acc=",rf.score(x_test,y_test))

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

base_models=[
    ("dt",DecisionTreeClassifier(random_state=42)),
    ('lr', LogisticRegression(random_state=42)),
    ('knn', KNeighborsClassifier())

]

meata_modell=LogisticRegression()
stacking=StackingClassifier(estimators=base_models,
                            final_estimator=meata_modell)
stacking.fit(x_train,y_train)
print("Stacking准确率:", stacking.score(x_test, y_test))