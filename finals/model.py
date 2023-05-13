import os

import matplotlib.pyplot as plt

# from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# from sklearn.model_selection import train_test_split


def modeling(df):

    # Step 1: Sort DataFrame based on the year column
    df_sorted = df.sort_values("year")
    print(df_sorted[["year", "home_team_wins"]])
    # Step 2: Determine the cutoff year for splitting the data
    cutoff_year = df_sorted["year"].max() - 1
    print("cutoff_year:", cutoff_year)

    # Step 3: Split DataFrame into X (features) and y (target variable)
    X = df_sorted.drop("home_team_wins", axis=1)
    y = df_sorted["home_team_wins"]

    # Step 4: Identify the index where the cutoff year starts
    cutoff_index = (X["year"] == cutoff_year).idxmax()
    print("cutoff_index", cutoff_index)
    # Step 5: Create the training and testing sets based on the cutoff index
    X_train = X.loc[:cutoff_index]
    X_test = X.loc[cutoff_index:]
    y_train = y.loc[:cutoff_index]
    y_test = y.loc[cutoff_index:]

    # Step 6: reset the index for X_train, X_test, y_train, and y_test
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    print("y_train:", y_train)
    y_test.reset_index(drop=True, inplace=True)

    # Define the output directory
    output_dir = "roc_curves_and_outliers"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    # Calculate ROC curve and AUC score
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="Logistic Regression (AUC = {:.2f})".format(auc_score))
    plt.plot([0, 1], [0, 1], "k--")
    # plt.scatter(fpr, tpr, c=thresholds, cmap='viridis')
    # plt.colorbar(label='Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - Logistic Regression")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "Logistic_Regression_ROC.png"))
    plt.close()

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
    # Calculate ROC curve and AUC score
    y_pred_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="Random Forest (AUC = {:.2f})".format(auc_score))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - Random Forest")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "Random_Forest_ROC.png"))
    plt.close()

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

    # Calculate ROC curve and AUC score
    y_pred_prob = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="KNN (AUC = {:.2f})".format(auc_score))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - K-Nearest Neighbors")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "KNN_ROC.png"))
    plt.close()

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

    # Calculate ROC curve and AUC score
    y_pred_prob = gb.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="Gradient Boosting (AUC = {:.2f})".format(auc_score))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - Gradient Boosting")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "Gradient_Boosting_ROC.png"))
    plt.close()

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
    # Calculate ROC curve and AUC score
    y_pred_prob = xgb.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="XGBoost (AUC = {:.2f})".format(auc_score))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - XGBoost")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "XGBoost_ROC.png"))
    plt.close()

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
    # Calculate ROC curve and AUC score
    y_pred_prob = dt.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="Decision Tree (AUC = {:.2f})".format(auc_score))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - Decision Tree")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "Decision_Tree_ROC.png"))
    plt.close()
