'''
    任务:鸢尾花的识别
'''
# -*- coding: utf-8 -*-
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 使用 pathlib 获取脚本目录，跨平台更好
DATA_FILE = Path(__file__).parent / 'data' / 'iris.csv'

SPECIES = {'setosa': 0,      # 山鸢尾
           'versicolor': 1,  # 变色鸢尾
           'virginica': 2}   # 维吉利亚鸢尾

# 使用特征列
FEAT_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
def main():
    # 读取数据集
    iris_data = pd.read_csv(DATA_FILE)
    iris_data['Label'] = iris_data['species'].map(SPECIES)
    # 获取数据特征
    X=iris_data[FEAT_COLS].values
    # 获取数据标签
    y=iris_data['Label'].values
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10, stratify=y)

    model_dict={'knn':(KNeighborsClassifier(),{'n_neighbors':[5,15,25],'p':[1,2]}),
                'log_reg': (LogisticRegression(max_iter=1000),{'C':[1e-3,1,1e2]}),
                'svm':(SVC(),{'C':[1e-3,1,1e2]}),
                'dt': (DecisionTreeClassifier(),{'max_depth':[3,6,9]})}

    # 测试并打印精度
    for model_name, (model, param_grid) in model_dict.items():
        #训练模型
        clf = GridSearchCV(model, param_grid, cv=5)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        #验证模型
        accuracy = best_model.score(X_test, y_test)
        #打印精度
        print(f'{model_name} (最佳参数: {clf.best_params_}) 精确度：{accuracy * 100:.2f}%')

if __name__ == '__main__':
     main()