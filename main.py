import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

X_train = None
X_test = None
y_train = None
y_test = None


def read_dataset():
    return {'data': pd.read_csv('dataset/data.csv').drop(columns=['Unnamed: 0']),
            'labels': pd.read_csv('dataset/labels.csv').drop(columns=['Unnamed: 0'])}


def normalizer():
    global X_train, X_test
    X_train = MinMaxScaler().fit_transform(X_train, y_train)
    X_test = MinMaxScaler().fit_transform(X_test, y_test)
    return


def data_preprocessing_with_feature_selection():
    dataset = read_dataset()
    X = dataset['data']
    y = dataset['labels']
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    normalizer()
    features = list(X.columns)
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    feature_scores_X_train = pd.concat(
        [pd.DataFrame(X.columns), pd.DataFrame(SelectKBest(score_func=chi2, k=50).fit(X_train, y_train).scores_)],
        axis=1)
    feature_scores_X_train.columns = ['Feature', 'Score']
    best_features = feature_scores_X_train.nlargest(50, 'Score')['Feature'].values
    X_train = X_train[best_features]
    X_test = X_test[best_features]

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values.reshape(640, )
    y_test = y_test.values.reshape(161, )
    return


def data_preprocessing_with_feature_extraction():
    dataset = read_dataset()
    X = dataset['data']
    y = dataset['labels']
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    normalizer()
    y_train = y_train.values.reshape(640, )
    y_test = y_test.values.reshape(161, )
    lda = LinearDiscriminantAnalysis()
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    return


def train_predict():
    print('Logistic Regression')
    lr = LogisticRegression(solver='saga', random_state=42, max_iter=700)
    lr.fit(X_train, y_train)
    # print(lr.decision_function(X_test))
    y_pred_test = lr.predict(X_test)
    print('Test set confusion matrix:')
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    print('Accuracy:')
    print(cm.trace() / cm.sum())
    print()
    y_pred_train = lr.predict(X_train)
    print('Train set confusion matrix:')
    cm = confusion_matrix(y_train, y_pred_train)
    print(cm)
    print('Accuracy:')
    print(cm.trace() / cm.sum())

    print()
    print()

    print('Gaussian Naive Bayes')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # print(gnb.class_prior_)
    y_pred_test = lr.predict(X_test)
    print('Test set confusion matrix:')
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    print('Accuracy:')
    print(cm.trace() / cm.sum())
    print()
    y_pred_train = lr.predict(X_train)
    print('Train set confusion matrix:')
    cm = confusion_matrix(y_train, y_pred_train)
    print(cm)
    print('Accuracy:')
    print(cm.trace() / cm.sum())
    return


if __name__ == '__main__':
    # data_preprocessing_with_feature_selection()
    data_preprocessing_with_feature_extraction()
    train_predict()
