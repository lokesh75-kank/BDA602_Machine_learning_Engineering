from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# from sklearn.model_selection import train_test_split


def modeling(df):

    # Step 1: Sort DataFrame based on the year column
    df_sorted = df.sort_values("year")

    # Step 2: Determine the cutoff year for splitting the data
    cutoff_year = df_sorted["year"].max() - 1
    print("cutoff_year:", cutoff_year)

    # Step 3: Split DataFrame into X (features) and y (target variable)
    X = df_sorted.drop("home_team_wins", axis=1)
    y = df_sorted["home_team_wins"]

    # Step 4: Identify the index where the cutoff year starts
    cutoff_index = (X["year"] == cutoff_year).idxmax()

    # Step 5: Create the training and testing sets based on the cutoff index
    X_train = X.loc[:cutoff_index]
    X_test = X.loc[cutoff_index:]
    y_train = y.loc[:cutoff_index]
    y_test = y.loc[cutoff_index:]

    # Step 6: reset the index for X_train, X_test, y_train, and y_test
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

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

    # KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print("Accuracy of KNN", accu)
    print(
        "Classification Report of K-Nearest Neighbors\n",
        classification_report(y_test, y_pred),
    )

    # Gradient Boosting model
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print("Accuracy of Gradient Boosting", accu)
    print(
        "Classification Report of Gradient Boosting Classifier\n",
        classification_report(y_test, y_pred),
    )

    # XGBoost model
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print("Accuracy of XGBoost", accu)
    print(
        "Classification Report of XGBoost Classifier\n",
        classification_report(y_test, y_pred),
    )

    # Decision Tree model
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print("Accuracy of Decision Tree", accu)
    print(
        "Classification Report of Decision Tree Classifier\n",
        classification_report(y_test, y_pred),
    )
