import numpy as np
from sklearn import tree
from data_utils import build_dataset
from c45_tree import DecisionTree

def main():
    train_x, train_y, test_x, test_y, feat_ranges, feat_names = build_dataset()

    model = DecisionTree(lbd=1.0).fit(train_x, train_y, feat_ranges)
    print(f'[手写 C4.5] 叶节点数: {model.T}')
    print(f'[手写 C4.5] 训练准确率: {model.accuracy(train_x, train_y):.4f}')
    print(f'[手写 C4.5] 测试准确率: {model.accuracy(test_x, test_y):.4f}')

    c45 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)
    cart = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)
    c45.fit(train_x, train_y)
    cart.fit(train_x, train_y)

    print(f'[Sklearn C4.5] 训练准确率: {c45.score(train_x, train_y):.4f}')
    print(f'[Sklearn C4.5] 测试准确率: {c45.score(test_x, test_y):.4f}')
    print(f'[Sklearn CART] 训练准确率: {cart.score(train_x, train_y):.4f}')
    print(f'[Sklearn CART] 测试准确率: {cart.score(test_x, test_y):.4f}')

if __name__ == '__main__':
    main()