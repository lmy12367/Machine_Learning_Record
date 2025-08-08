from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_util import  get_clf_data
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"   

x_train,X_test,y_train,y_test=get_clf_data()

stump=DecisionTreeClassifier(max_depth=1,random_state=42)

M=np.arange(1,101,5)

scores=[]

for m in M:
    ada = AdaBoostClassifier(
        base_estimator=stump,
        n_estimators=m,
        algorithm='SAMME',
        random_state=42
    )
    ada.fit(x_train, y_train)
    scores.append(accuracy_score(y_test, ada.predict(X_test)))

plt.plot(M, scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.title("AdaBoost with stumps")
plt.tight_layout()
plt.savefig("03_adaboost.png")
plt.show()

print("AdaBoost(100):", scores[-1])