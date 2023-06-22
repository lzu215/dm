# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Random forest
def random_forest_classify(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


# AdaBoost
def ada_boost_classify(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


# Stacking
def stacking_classify(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    estimators = [
        ('knn', KNeighborsClassifier(n_neighbors=3)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(probability=True, kernel='linear'))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)


# standardize and split data
def standardize_split(x, y):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=32)
    return x_train, x_test, y_train, y_test


# visualization
def visualize(df: pd.DataFrame):
    plt.ylim((0.8, 1.1))
    sns.barplot(data=df, x=df.columns[0], y=df.columns[1], width=0.5, palette='Blues')
    plt.title('Accuracy of Classifiers')
    plt.tight_layout()
    plt.savefig('../figure/accuracy.svg', dpi=600, format='svg')
    plt.show()


# main function
def run(df: pd.DataFrame):
    results = []
    x, y = df.iloc[:, 1:], df.iloc[:, 0]
    x_train, x_test, y_train, y_test = standardize_split(x, y)
    models = {'Random Forest': random_forest_classify, 'AdaBoost': ada_boost_classify, 'Stacking': stacking_classify}
    for name, model in models.items():
        results.append([name, model(x_train, x_test, y_train, y_test)])
    results = pd.DataFrame(results, columns=['Classifier', 'Accuracy'])
    visualize(results)
    print(results)


if __name__ == '__main__':
    cancer_data = pd.read_csv('../data/Cancer_Data.csv', sep=',', index_col=0).iloc[:, :-1]
    run(cancer_data)
