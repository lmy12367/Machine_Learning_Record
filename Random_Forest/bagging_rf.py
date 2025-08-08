from tkinter.tix import Tree
from sklearn import tree
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_util import get_clf_data
import matplotlib.pyplot as plt
import numpy as np

x_train,x_test,y_trian,y_test=get_clf_data()

tree=DecisionTreeClassifier(random_state=42)
tree.fit(x_train,y_trian)
print("单棵决策树:", accuracy_score(y_test, tree.predict(x_test)))

bag=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    oob_score=True,
    random_state=42
)
bag.fit(x_train,y_trian)
print("Bagging   :", accuracy_score(y_test, bag.predict(x_test)))
print("Bagging OOB:", bag.oob_score_)

rf=RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    oob_score=True,
    random_state=42
)
rf.fit(x_train,y_trian)
print("随机森林   :", accuracy_score(y_test, rf.predict(x_test)))
print("随机森林OOB:", rf.oob_score_)

trees=np.arange(1,101,5)
bag_scores,rf_scores=[],[]

for n in trees:
    bag_scores.append(
        BaggingClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n,
            random_state=42
        ).fit(x_train,y_trian).score(x_test,y_test)
    )
    
    rf_scores.append(
        RandomForestClassifier(
            n_estimators=n,
            max_features='sqrt',
            random_state=42
        ).fit(x_train,y_trian).score(x_test,y_test)
    )
    
plt.plot(trees, bag_scores, label='Bagging')
plt.plot(trees, rf_scores, label='RandomForest')
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.legend()
plt.title("Bagging vs RandomForest")
plt.tight_layout()
plt.savefig("01_bagging_rf.png")
plt.show()