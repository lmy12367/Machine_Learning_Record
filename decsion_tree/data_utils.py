import numpy as np
import pandas as pd

def build_dataset(ratio: float = 0.8, bins: int = 10, seed: int = 0):

    data = pd.read_csv('./data/ml/decsion_tree/titanic/train.csv')


    data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)


    data.fillna(-1, inplace=True)


    feat_ranges = {}
    cont_feat = ['Age', 'Fare']
    for feat in cont_feat:
        min_val, max_val = data[feat].min(), data[feat].max()
        feat_ranges[feat] = [-1] + np.linspace(min_val, max_val, bins).tolist()


    cat_feat = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
    for feat in cat_feat:
        data[feat] = data[feat].astype('category').cat.codes
        feat_ranges[feat] = [-1] + sorted(data[feat].unique())

    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(ratio * len(data))
    train_df, test_df = data[:split], data[split:]

    train_x = train_df.drop(columns=['Survived']).to_numpy()
    train_y = train_df['Survived'].to_numpy()
    test_x = test_df.drop(columns=['Survived']).to_numpy()
    test_y = test_df['Survived'].to_numpy()

    feat_names = list(train_df.columns.drop('Survived'))
    return train_x, train_y, test_x, test_y, feat_ranges, feat_names