import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score


# Stacking
def stacking_classify(x_train, x_test, y_train, y_test):
    estimators = [
        ('knn', KNeighborsClassifier(n_neighbors=3)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(probability=True, kernel='linear'))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy of Stacking: ', accuracy_score(y_test, y_pred))


# 标准化划分函数
def standardize_split(x, y):
    scaler = StandardScaler()  # 数据标准化
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    return x_train, x_test, y_train, y_test


# 主调函数
def run(df: pd.DataFrame):
    x, y = df.iloc[:, 1:], df.iloc[:, 0]
    x_train, x_test, y_train, y_test = standardize_split(x, y)
    stacking_classify(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    breast_cancer_wisconsin = pd.read_table('../data/breast_cancer_wisconsin/wdbc.data',
                                            sep=',', index_col=0, header=None)
    run(breast_cancer_wisconsin)
