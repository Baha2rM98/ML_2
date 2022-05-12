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


# We read the data and labels and return them in a dictionary to be used later
def read_dataset():
    return {'data': pd.read_csv('dataset/data.csv').drop(columns=['Unnamed: 0']),
            'labels': pd.read_csv('dataset/labels.csv').drop(columns=['Unnamed: 0'])}


# We normalize X_train and X_test set with min-max method
def normalizer():
    global X_train, X_test
    X_train = MinMaxScaler().fit_transform(X_train, y_train)
    X_test = MinMaxScaler().fit_transform(X_test, y_test)
    return


# We preprocess the data to be optimized for training our model with univariate feature selection method
def data_preprocessing_with_feature_selection():
    # First we read the dataset
    dataset = read_dataset()
    X = dataset['data']
    y = dataset['labels']

    # Splitting the dataset into X_train, X_test, y_train and y_test set
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # We normalize X_train and X_test set
    normalizer()

    # Selecting 25 of best features of the dataset to reduce data dimension
    features = list(X.columns)
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    feature_scores_X_train = pd.concat(
        [pd.DataFrame(features), pd.DataFrame(SelectKBest(score_func=chi2, k=25).fit(X_train, y_train).scores_)],
        axis=1)
    feature_scores_X_train.columns = ['Feature', 'Score']
    best_features = feature_scores_X_train.nlargest(25, 'Score')['Feature'].values
    X_train = X_train[best_features]
    X_test = X_test[best_features]

    # Convert dataframe into numpy array
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values.reshape(640, )
    y_test = y_test.values.reshape(161, )
    return


# We preprocess the data to be optimized for training our model with LDA feature extraction method
def data_preprocessing_with_feature_extraction():
    # First we read the dataset
    dataset = read_dataset()
    X = dataset['data']
    y = dataset['labels']

    # Splitting the dataset into X_train, X_test, y_train and y_test set
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # We normalize X_train and X_test set
    normalizer()

    # Converting y_train and y_test into numpy arrays then fit the LDA algorithm with X_train and y_train then reduce
    # the dimension to Four features.
    y_train = y_train.values.reshape(640, )
    y_test = y_test.values.reshape(161, )
    lda = LinearDiscriminantAnalysis(n_components=4)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    return


def train_predict():
    print('Logistic Regression')
    lr = LogisticRegression(solver='saga', random_state=42, max_iter=700)
    lr.fit(X_train, y_train)
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
    print('--------------------------------')
    print()

    print('Gaussian Naive Bayes')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_test = gnb.predict(X_test)
    print('Test set confusion matrix:')
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    print('Accuracy:')
    print(cm.trace() / cm.sum())
    print()
    y_pred_train = gnb.predict(X_train)
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
