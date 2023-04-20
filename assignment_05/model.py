from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def modeling(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=20
    )
    # LogisticRegression model
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accu = accuracy_score(y_test, y_predict)
    print("Accuracy of Logistic Regression", accu)
    print(
        "Classification Report of Logistic Regression\n",
        classification_report(y_test, y_predict),
    )
    # support vector machine
    model = svm.SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print("Accuracy of SVM", accu)
    print(
        "Classification Report of Support Vector Classifier\n",
        classification_report(y_test, y_pred),
    )

    # Random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print("Accuracy of Random_Forest", accu)
    print(
        "Classification Report of Random forest Classifier\n",
        classification_report(y_test, y_pred),
    )
