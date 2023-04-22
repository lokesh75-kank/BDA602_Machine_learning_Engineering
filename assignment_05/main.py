# Importing relevant packages
import sys
import warnings

# import pandas as pd
# import sqlalchemy
from model import modeling
from pyspark import StorageLevel
from pyspark.sql import SparkSession

from assignment_05.brute_force import (
    cat_cat_brute_force,
    cat_con_brute_force,
    con_con_brute_force,
)
from assignment_05.calculate_correlation_metrics import create_corrheatmapfigs
from assignment_05.creating_html import creating_html
from assignment_05.predictors_tables import create_predictor_dfs, predictor_plots

# Ignore FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)


def main():
    # database properties
    database = "baseball"
    user = "admin"
    password = "1196"
    server = "localhost"
    port = 3306
    # db = "baseball"
    # host = "localhost:3306"

    # Tried using sqlalchemy to connect DB #
    """ I tried both ways for connecting DB.
        The query runs fast with spark also few plots are not
        generating with sqlalchemy hence using spark for DB connection
    """
    # SQL queries
    # c = f"mariadb+mariadbconnector://{user}:{password}@{host}/{db}"  # pragma: allowlist secret
    # query = "SELECT * FROM feature_ratio"
    # sql_engine = sqlalchemy.create_engine(c)
    # df = pd.read_sql_query(query, sql_engine)
    # print(df.head(5))

    # connected using pyspark as is runs fast on my machine#
    appName = "PySpark Example - MariaDB Example"
    master = "local[*]"
    # Create Spark session
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"
    sql1 = """
                Select * from feature_ratio
            """
    # Create a data frame by reading data from Oracle via JDBC using sql1
    df_sql1 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql1)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )
    df_sql1.createOrReplaceTempView("feature_ratio")
    df_sql1.persist(StorageLevel.DISK_ONLY)
    # df_sql1.show(1)
    df = df_sql1.toPandas()

    df = df.drop(["game_id", "home_team_id", "away_team_id"], axis=1)
    # # print(df.isnull().sum())
    for col in df.columns:
        df[col].fillna(0, inplace=True)

    # modeling part
    X = df.drop("home_team_wins", axis=1)
    y = df["home_team_wins"]
    modeling(X, y)
    print(
        "Model Comparision:\n"
        "Based on the given results, it appears that the logistic regression model has the highest accuracy "
        "with a score of 0.5569, followed by the support vector machine (SVM) with an accuracy of 0.5421, "
        "and the random forest classifier with an accuracy of 0.5242.\n"
        "Looking at the classification reports, the logistic regression and SVM models have relatively high "
        "precision and recall for class 1, which suggests that they are performing well in identifying instances "
        "of that class. However, they have low precision and recall for class 0, which suggests that they are "
        "not performing as well in identifying instances of that class.\n"
        "The random forest classifier, on the other hand, has relatively balanced precision and recall for both "
        "classes, but its overall accuracy is lower than the other models."
    )

    # Feature Analyzing
    # code takes time to run for all records for plots so kept df with 10 rows.

    df = df.head(10)
    predictors = df.columns[:-1]
    response = "home_team_wins"
    df_pred = df[predictors]
    # creating dataframe of the columns with names from the response
    df_res = df[response]
    # print(df_res.columns)

    # df_res= df_res.astype('object')

    # this function saves all the plts in the directories dynamically, also returns
    # list of continuous predictors and catagorical predictors
    continuous_pred, catagorical_pred = predictor_plots(df_pred, df_res)
    # print(continuous_pred,"\n",catagorical_pred)
    # # function gives complete 2 dfs to convert to html
    continuous_pred_df, catagorical_pred_df = create_predictor_dfs(
        df, response, continuous_pred, catagorical_pred
    )
    #
    # # BRUTE FORCE TABLES
    # # gives complete brute force dfs(with and without dropping the same column names)
    #
    # """I tried calculations for the diff_mean_resp_ranking, diff_mean_resp_weighted_ranking
    # in the brute force table.The calculations for categorical/categorical are accurate,
    # but somehow the values for other two tables are not accurate.
    # I have also tried plotting the heatmaps.They are accurate but,I couldn't linked them to the dataframe
    # because i got few errors.Hence, I have stored them in a folder. """
    cat_cat_brute_force_df, cat_cat_complete = cat_cat_brute_force(
        catagorical_pred, df, response
    )
    cat_con_brute_force_df, cat_con_complete = cat_con_brute_force(
        catagorical_pred, continuous_pred, df, response
    )
    con_con_brute_force_df, con_con_complete = con_con_brute_force(
        continuous_pred, df, response
    )
    #
    # # CODE FOR CORRELATION 3 CORRELATION MATRICES AND 3 PLOTS
    # # extract desired columns
    corr_cat_cat = cat_cat_complete[["cat_1", "cat_2", "pearson"]]
    corr_con_cat = cat_con_complete[["cat", "cont", "corr_ratio"]]
    corr_con_con = con_con_complete[["cont_1", "cont_2", "pearson"]]

    # # create their heatmaps
    corr_figs = create_corrheatmapfigs(corr_cat_cat, corr_con_cat, corr_con_con)
    # df_con_pred = df[continuous_pred]
    # # impurity_based_feature_importance(df, df_con_pred, response)
    # # converts all the dataframes to html
    dataset_name = "Baseball"
    creating_html(
        dataset_name,
        corr_cat_cat,
        corr_con_cat,
        corr_con_con,
        corr_figs,
        continuous_pred_df,
        catagorical_pred_df,
        cat_cat_brute_force_df,
        cat_con_brute_force_df,
        con_con_brute_force_df,
    )


if __name__ == "__main__":
    sys.exit(main())
