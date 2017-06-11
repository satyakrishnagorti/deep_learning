from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data_set(file_name):
    dataaset = pd.read_csv(file_name)
    X = dataaset.iloc[:, 3:13].values
    y = dataaset.iloc[:, 13].values
    return X, y


def tranform_data(X, y):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # encoding countries as integers
    label_encoder_country = LabelEncoder()
    X[:, 1] = label_encoder_country.fit_transform(X[:, 1])

    # encoding gender Male, Female -> 0,1
    label_encoder_gender = LabelEncoder()
    X[:, 2] = label_encoder_gender.fit_transform(X[:, 2])

    # one hot encoding the categorical feature geography
    onehotencoder = OneHotEncoder(categorical_features=[1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]

    return X, y


def feature_scale(X_train, X_test):
    # (x - mean) / stddev
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


def ttsplit(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def build_classifier():

    # Network dim-> (11, 6, 6, 1)
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


def main():
    X, y = read_data_set("Churn_Modelling.csv")
    X, y = tranform_data(X, y)
    X_train, X_test, y_train, y_test = ttsplit(X, y)
    X_train, X_test = feature_scale(X_train, X_test)
    
    classifier = build_classifier()

    classifier.fit(X_train, y_train, batch_size=10, epochs=50)

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    print(cm)
    """
    [[1525   70]
    [ 209  196]]

    Accuracy: ~ 86%
    """


if __name__ == '__main__':
    main()
