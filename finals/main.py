# Importing relevant packages
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brute_force import cat_cat_brute_force, cat_con_brute_force, con_con_brute_force
from calculate_correlation_metrics import create_corrheatmapfigs
from creating_html import creating_html
from model import modeling
from predictors_tables import (
    create_predictor_dfs,
    impurity_based_feature_importance,
    predictor_plots,
)
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

# Ignore FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


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

    query = "SELECT * FROM baseball_final_features"
    df = pd.read_sql(query, engine)

    print("Features_analyzing")

    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(0, inplace=True)

    df = df.drop(
        columns=[
            "bag_game_id",
            "game_id_ratio",
            "home_team_id_ratio",
            "away_team_id",
            "common_team_id",
            "common_game_id",
            "HBP_ISO_Ratio",
        ]
    )
    response = "home_team_wins"
    df_pred = df.drop("home_team_wins", axis=1)

    # # creating dataframe of the columns with names from the response
    df_res = df[response]

    # # this function saves all the plts in the directories dynamically, also returns
    # # list of continuous predictors and catagorical predictors
    print("nunique strikout_per_inn:", df["strikout_per_inn"].unique())
    continuous_pred, catagorical_pred = predictor_plots(df_pred, df_res)

    # function gives complete 2 dfs to convert to html
    continuous_pred_df, catagorical_pred_df = create_predictor_dfs(
        df, response, continuous_pred, catagorical_pred
    )
    print("Running BRUTE FORCE cat cat TABLES")

    cat_cat_brute_force_df, cat_cat_complete = cat_cat_brute_force(
        catagorical_pred, df, response
    )
    print("Running BRUTE FORCE cat con TABLES")

    cat_con_brute_force_df, cat_con_complete = cat_con_brute_force(
        catagorical_pred, continuous_pred, df, response
    )
    print("Running BRUTE FORCE con con TABLES")
    con_con_brute_force_df, con_con_complete = con_con_brute_force(
        continuous_pred, df, response
    )

    # CODE FOR CORRELATION 3 CORRELATION MATRICES AND 3 PLOTS
    # extract desired columns
    corr_cat_cat = cat_cat_complete[["cat_1", "cat_2", "pearson"]]
    corr_con_cat = cat_con_complete[["cat", "cont", "corr_ratio"]]
    corr_con_con = con_con_complete[["cont_1", "cont_2", "pearson"]]

    # # create their heatmaps
    corr_figs = create_corrheatmapfigs(corr_cat_cat, corr_con_cat, corr_con_con)
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

    print("Features_engineering")

    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Reshape the column to a 2-dimensional array (required by MinMaxScaler)
    column_data = df["strikout_per_inn"].values.reshape(-1, 1)

    # Scale the values between 0 and 1
    scaled_values = scaler.fit_transform(column_data)

    # Assign the scaled values back to the DataFrame
    df["scaled_strikout_per_inn"] = scaled_values

    def visualize_outliers(df):
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot box plots for each column
        sns.boxplot(data=df, ax=ax)

        # Set plot labels
        ax.set_xticklabels(df.columns, rotation=90)
        ax.set_ylabel("Values")
        # Save the figure
        plt.savefig("roc_curves_and_outliers/outliers.png")

        # Show the plot
        plt.show()

    # visualize_outliers before
    visualize_outliers(df)

    def handle_outliers(df, method="median", threshold=1.5):

        # Copy the DataFrame to avoid modifying the original data
        df_cleaned = df.copy()
        df_cleaned = pd.DataFrame(df_cleaned)

        # Iterate over each column in the DataFrame
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype != np.number:
                continue  # Skip non-numeric columns

            # Calculate the lower and upper bounds for outliers
            if method == "median":
                median = df_cleaned[col].median()
                diff = threshold * (
                    df_cleaned[col].quantile(0.75) - df_cleaned[col].quantile(0.25)
                )
                lower_bound = median - diff
                upper_bound = median + diff
            elif method == "mean":
                mean = df_cleaned[col].mean()
                std = df_cleaned[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                raise ValueError("Invalid method. Choose either 'median' or 'mean'.")

            # Identify and handle outliers
            outliers = df_cleaned[
                (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            ]
            if not outliers.empty:
                if method == "median":
                    df_cleaned.loc[outliers.index, col] = np.clip(
                        df_cleaned.loc[outliers.index, col], lower_bound, upper_bound
                    )
                elif method == "mean":
                    df_cleaned.loc[outliers.index, col] = df_cleaned[col].apply(
                        lambda x: lower_bound
                        if x < lower_bound
                        else (upper_bound if x > upper_bound else x)
                    )

        return df_cleaned

    df = handle_outliers(df, method="median", threshold=1.5)

    # visualize_outliers AFTER
    # visualize_outliers(df)

    modeling(df)

    impurity_based_feature_importance(df, response)

    print(
        """
        After comparing the models, here's a brief summary of the results:
Logistic Regression has the highest accuracy (0.546) and a relatively high recall (0.962),
indicating that it correctly identifies a large proportion of positive instances.
Random Forest has lower accuracy (0.500) but shows balanced precision (0.535) and
recall (0.646) scores.
KNN has moderate accuracy (0.523) and performs reasonably well in terms of precision
(0.553) and recall (0.659).
Gradient Boosting has a relatively high recall (0.812), indicating it identifies a
higher proportion of positive instances, but has lower accuracy (0.541).
XGBoost has similar performance to the other models with moderate accuracy (0.520)
and precision (0.552) scores.
Decision Tree has the lowest accuracy (0.505) and relatively low precision (0.548)
and recall (0.540) scores.
Based on these results, the Logistic Regression model appears to be the best overall,
as it has the highest accuracy and reasonably balanced precision and recall scores.
    """
    )


if __name__ == "__main__":
    sys.exit(main())
