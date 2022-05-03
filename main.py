import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

X_train = None
X_test = None
y_train = None
y_test = None


def read_dataset():
    return {'data': pd.read_csv('dataset/data.csv').drop(columns=['Unnamed: 0']),
            'labels': pd.read_csv('dataset/labels.csv').drop(columns=['Unnamed: 0'])}


def feature_selector():
    dataset = read_dataset()
    X = dataset['data']
    y = dataset['labels']
    feature_scores = pd.concat(
        [pd.DataFrame(X.columns), pd.DataFrame(SelectKBest(score_func=chi2, k=50).fit(X, y).scores_)],
        axis=1)
    feature_scores.columns = ['Features', 'Score']
    best_features = X[feature_scores.nlargest(50, 'Score')['Features'].values]
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(best_features.values, y.values.reshape(801, ), test_size=0.2,
                                                        random_state=42)
    # df = pd.DataFrame(y_test)
    # df.at[0, 0] = 'BRCA'
    # y_test = df.values.reshape(161, )
    # print(y_test)
    return


def feature_extractor():
    dataset = read_dataset()
    X = dataset['data'].values
    y = dataset['labels'].values.reshape(801, )
    lda = LDA()
    lda.fit(X, y)
    return lda.transform(X).shape


def normalizer():
    global X_train, X_test
    X_train = MinMaxScaler().fit_transform(X_train, y_train)
    X_test = MinMaxScaler().fit_transform(X_test, y_test)
    return


def fit_train_predict():
    print('Logistic Regression')
    lr = LogisticRegression(solver='saga', random_state=42, multi_class='multinomial')
    lr.fit(X_train, y_train)
    y_pred_test = lr.predict(X_test)
    print('Test confusion matrix:')
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    print()
    print('Accuracy:')
    print(cm.trace() / cm.sum())
    print()
    print('Classification report:')
    print(classification_report(y_test, y_pred_test))
    print()
    y_pred_train = lr.predict(X_train)
    print('Train confusion matrix:')
    cm = confusion_matrix(y_train, y_pred_train)
    print(cm)
    print()
    print('Accuracy:')
    print(cm.trace() / cm.sum())
    print()
    print('Classification report:')
    print(classification_report(y_train, y_pred_train))

    print()
    print()

    print('Gaussian Naive Bayes')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_test = lr.predict(X_test)
    print('Test confusion matrix:')
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    print()
    print('Accuracy:')
    print(cm.trace() / cm.sum())
    print()
    print('Classification report:')
    print(classification_report(y_test, y_pred_test))
    print()
    y_pred_train = lr.predict(X_train)
    print('Train confusion matrix:')
    cm = confusion_matrix(y_train, y_pred_train)
    print(cm)
    print()
    print('Accuracy:')
    print(cm.trace() / cm.sum())
    print()
    print('Classification report:')
    print(classification_report(y_train, y_pred_train))
    return


if __name__ == '__main__':
    # print(feature_extractor())
    feature_selector()
    normalizer()
    fit_train_predict()
