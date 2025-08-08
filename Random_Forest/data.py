from tkinter import Radiobutton
from sklearn.datasets import make_classification,make_friedman1
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_STATE=42

def get_clf_data():
    x,y=make_classification(
        n_samples=1000,n_features=16,n_informative=5,
        n_redundant=2,n_classes=2,flip_y=0.1,
        random_state=RANDOM_STATE
    )

    return train_test_split(x,y,test_size=0.2,random_state=RANDOM_STATE)

def get_reg_data():
    x,y=make_friedman1(
        n_samples=2000,n_features=100,noise=0.5,
        random_state=RANDOM_STATE
    )

    return train_test_split(x,y,test_size=0.2,random_state=RANDOM_STATE)
