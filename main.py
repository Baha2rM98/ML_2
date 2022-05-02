import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

X_train = None
X_test = None
y_train = None
y_test = None


def read_dataset() -> dict:
    return {'data': pd.read_csv('dataset/data.csv').drop(columns=['Unnamed: 0']),
            'labels': pd.read_csv('dataset/labels.csv').drop(columns=['Unnamed: 0'])}


def dataset_splitter() -> None:
    dataset = read_dataset()
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['labels'], test_size=0.2,
                                                        random_state=42)
    return


def feature_selector() -> None:
    # apply SelectKBest class to extract top 10 best features
    dataset = read_dataset()
    X = dataset['data']
    # y = LabelEncoder().fit_transform(dataset['labels'])
    y = dataset['labels']
    best_features = SelectKBest(score_func=chi2, k=10)
    scores = pd.DataFrame(best_features.fit(X, y).scores_)
    columns = pd.DataFrame(X.columns)
    featureScores = pd.concat([columns, scores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print(featureScores.nlargest(10, 'Score'))
    return


if __name__ == '__main__':
    feature_selector()
