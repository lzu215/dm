import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score


# 随机森林
def random_forest_classify(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


# AdaBoost
def ada_boost_classify(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    model = AdaBoostClassifier()  # 基分类器默认采用决策树模型
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def visualize(df: pd.DataFrame):
    plt.ylim((0.5, 1.1))
    sns.barplot(data=df, x=df.columns[0], y=df.columns[1], hue=df.columns[2], width=0.5, palette='Blues')
    plt.title('Accuracy of Classifiers')
    plt.tight_layout()
    plt.savefig('../figure/accuracy_classification.svg', dpi=600, format='svg')
    plt.show()


# 主调函数
def run(set1: tuple, set2: tuple):
    result, visual = [], []
    for name, dataset in {set1[0]: set1[1], set2[0]: set2[1]}.items():
        if name == 'Iris':
            x, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        else:
            x, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
        accuracy_1 = random_forest_classify(x_train, x_test, y_train, y_test)
        accuracy_2 = ada_boost_classify(x_train, x_test, y_train, y_test)
        result.append({'Random Forest': accuracy_1, 'AdaBoost': accuracy_2})
        visual.append([name, accuracy_1, 'Random Forest'])
        visual.append([name, accuracy_2, 'AdaBoost'])
    visualize(pd.DataFrame(visual, columns=['Name', 'Accuracy', 'Classifier']))
    accuracy = pd.DataFrame(result, index=[set1[0], set2[0]])
    accuracy.to_excel('../result/accuracy_classification.xlsx')
    print(accuracy)


if __name__ == '__main__':
    iris = pd.read_table('../data/iris/iris.data', sep=',', header=None)  # 读取数据)
    breast_cancer_wisconsin = pd.read_table('../data/breast_cancer_wisconsin/wdbc.data',
                                            sep=',', index_col=0, header=None)
    run(('Iris', iris), ('Breast cancer', breast_cancer_wisconsin))
