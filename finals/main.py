# Importing relevant packages
import sys
import warnings

# import numpy as np
import pandas as pd
from brute_force import cat_cat_brute_force, cat_con_brute_force, con_con_brute_force
from calculate_correlation_metrics import create_corrheatmapfigs
from creating_html import creating_html
from model import modeling
from predictors_tables import create_predictor_dfs, predictor_plots
from sqlalchemy import create_engine

# from pyspark import StorageLevel
# from pyspark.sql import SparkSession


# Ignore FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)


def main():
    # database properties
    database = "baseball"
    user = "root"
    password = "1196"
    server = "Lokesh_mariadb"
    port = 3306

    # Create the connection URL
    connection_url = f"mysql+pymysql://{user}:{password}@{server}:{port}/{database}"

    # Create the SQLAlchemy engine
    engine = create_engine(
        connection_url, echo=True
    )  # Set `echo` to True for verbose output

    # Test the connection
    try:
        connection = engine.connect()
        print("MariaDB Connected successfully!")
        connection.close()
    except Exception as e:
        print("MariaDB Connection failed:", e)

    query = "SELECT * FROM all_stats_features"
    df = pd.read_sql(query, engine)

    # print(type(df))
    # Tried using sqlalchemy to connect DB #
    """ I tried both ways for connecting DB."""

    # appName = "PySpark Example - MariaDB Example"
    # master = "local[*]"
    # # Create Spark session
    # spark = SparkSession.builder.appName(appName).master(master).getOrCreate()
    # jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    # jdbc_driver = "org.mariadb.jdbc.Driver"
    # sql1 = """
    #             Select * from Baseball_features
    #         """
    # # Create a data frame by reading data from Oracle via JDBC using sql1
    # df_sql1 = (
    #     spark.read.format("jdbc")
    #     .option("url", jdbc_url)
    #     .option("query", sql1)
    #     .option("user", user)
    #     .option("password", password)
    #     .option("driver", jdbc_driver)
    #     .load()
    # )
    # df_sql1.createOrReplaceTempView("feature_ratio")
    # df_sql1.persist(StorageLevel.DISK_ONLY)
    # # df_sql1.show(1)
    # df = df_sql1.toPandas()

    print("DATA PRE-PROCESSING")

    # Dropping unwanted columns
    df = df.drop(
        [
            "team_id",
            "player_id",
            "game_id_ratio",
            "bag_game_id",
            "home_team_id_ratio",
            "game_id",
            "away_team_id",
            "month_des",
            "loaded_hr",
            "team_des",
            "vs_b_des",
            "vs_b5_des",
        ],
        axis=1,
    )
    # print(df.columns)

    # After checking columns,am dropping those with complete null values
    df = df.drop(
        [
            "season_des",
            "career_des",
            "empty_des",
            "men_on_des",
            "risp_des",
            "vs_rhb_des",
            "loaded_des",
            "vs_lhb_des",
            "pitch_out",
        ],
        axis=1,
    )

    # null_counts_str = df.isnull().sum().to_string()
    # print(null_counts_str)

    # dtypes_col = df.dtypes.to_string()
    # print(dtypes_col)

    for col in df.columns:
        if df[col].isnull().any():
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)

    # Identify the columns to remove
    columns_to_remove = [col for col in df.columns if col.startswith("vs")]

    # Remove the columns from the DataFrame
    df = df.drop(columns=columns_to_remove)

    # Identifying the columns with zero varience to remove from the DataFrame
    columns_to_remove_zero_variance = [
        col for col in df.columns if df[col].nunique() == 1
    ]
    df = df.drop(columns=columns_to_remove_zero_variance)
    # print(df.columns.tolist())

    print("Feature Analyzing")
    df = df.head(10)
    # print("aetaaaaafda", df["home_team_wins"].info())
    # predictors = df.columns[:-1]
    response = "home_team_wins"
    df_pred = df.drop("home_team_wins", axis=1)

    # # creating dataframe of the columns with names from the response
    df_res = df[response]

    # # this function saves all the plts in the directories dynamically, also returns
    # # list of continuous predictors and catagorical predictors
    continuous_pred, catagorical_pred = predictor_plots(df_pred, df_res)
    # print(continuous_pred,"\n",catagorical_pred)
    # # function gives complete 2 dfs to convert to html
    continuous_pred_df, catagorical_pred_df = create_predictor_dfs(
        df, response, continuous_pred, catagorical_pred
    )

    print("Running BRUTE FORCE TABLES")

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

    print("data modeling part")

    # Compute the correlation matrix
    # corr_matrix = df.corr().abs()

    # # Create a mask to identify highly correlated features
    # upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    # highly_correlated = corr_matrix.mask(upper_triangle)
    #
    # # Find features with correlation above a certain threshold
    # threshold = 0.9  # Adjust this threshold as needed
    # collinear_features = set()
    # for feature in highly_correlated.columns:
    #     correlated = highly_correlated[feature][highly_correlated[feature] > threshold].index
    #     collinear_features.update(correlated)
    # print("ciafnalsdf",collinear_features)
    # # Remove the collinear features from the DataFrame
    # df = df.drop(columns=collinear_features)

    # Print the filtered DataFrame
    # print(df.info())

    modeling(df)

    print(
        """
        Here's a brief comparison of the accuracy and performance
        of the different models I have used:

        Logistic Regression:
        Accuracy: 0.516
        Precision, recall, and F1-score: Low for class 1, but high for class 0
        Overall, the model struggles to predict class 1 correctly.
        ==============================
        Support Vector Classifier (SVM):
        Accuracy: 0.548
        Precision, recall, and F1-score: Higher for class 1 than for class 0
        The model performs better than logistic regression, but still struggles with class 0 predictions.
        ========================
        Random Forest Classifier:
        Accuracy: 0.581
        Precision, recall, and F1-score: Balanced performance for both classes
        The model shows improvement over the previous models, but there is room for further improvement.
        ==========================
        K-Nearest Neighbors (KNN):
        Accuracy: 0.597
        Precision, recall, and F1-score: Better recall for class 0, but better precision for class 1
        The model shows a decent accuracy and performs reasonably well for both classes.
       ==============================
        Gradient Boosting Classifier:
        Accuracy: 0.597
        Precision, recall, and F1-score: Balanced performance for both classes
        The model performs similar to random forest classifier, achieving decent accuracy.
        ==================
        XGBoost Classifier:
        Accuracy: 0.565
        Precision, recall, and F1-score: Similar performance for both classes
        The model provides comparable results to the other models, but doesn't stand out significantly.
        ========================
        Decision Tree Classifier:
        Accuracy: 0.484
        Precision, recall, and F1-score: Balanced performance for both classes, but lower overall
        The model performs relatively poorly compared to the other models.
        ============================
        Overall, among the models evaluated, the K-Nearest Neighbors (KNN) and Gradient Boosting Classifier
        stand out with relatively better accuracy and performance."""
    )


if __name__ == "__main__":
    sys.exit(main())
