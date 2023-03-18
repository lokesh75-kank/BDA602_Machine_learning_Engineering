import sys
from statistics import statistics

# import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datasets import TestDatasets
from plotly import express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tabulate import tabulate


def main():
    # creating object of testdatasets class
    datasets = TestDatasets()
    # checking other datasets and selecting one
    # df, predictors, response = datasets.get_test_data_set("diabetes")
    # df, predictors, response = datasets.get_test_data_set("breast_cancer")
    df, predictors, response = datasets.get_test_data_set("boston")

    df["RAD"] = df["RAD"].astype("float64")
    """
    note: I dont know if am data manupulation is right way but I noticed
    that in reference graphs from slide that column type for
    "Rad" was continuous, also to train regression models
    data type "object" is not accepted hence converting it to "float".
    """

    # print(df.dtypes)
    predictors_df = df[predictors]
    # print(predictors_df)
    # print("col_dtype:",df.columns.dtype)

    def to_check_cat_con_pred(columns):
        dict = {}
        for col in columns:
            # print(col)
            if columns[col].dtype == "object":
                dict[col] = "== Categorical"
            elif columns[col].dtype in ["int64", "float64"]:
                dict[col] = "== Continuous"
        return dict

    print(to_check_cat_con_pred(predictors_df))

    def to_check_cat_con_response(column):
        for i in column:
            if isinstance(i, float):
                print("Continuous")
            else:
                print("Categorical")

    # print(to_check_cat_con_response(df["target"]))

    def ploting_graphs_cont_cat_target_predictors():
        for i in predictors_df:
            # print(i)
            x = i
            y = df["target"]
            fig = px.scatter(df, x=x, y=y, trendline="ols")
            if df["CHAS"].dtype == "object":
                fig.update_layout(
                    title="Continuous Response by Catagorical Predictor",
                    xaxis_title=f"Predictor_{i}",
                    yaxis_title=f"Response_{y}",
                )
            else:
                fig.update_layout(
                    title="Continuous Response by Continuous Predictor",
                    xaxis_title=f"Predictor_{i}",
                    yaxis_title=f"Response_{y}",
                )
            # fig.show()
            fig.write_html(
                file=f"plots_cont_cat/"
                f"lecture_6_cont_response_cont_predictor_"
                f"{i}_scatter_plot.html",
                include_plotlyjs="cdn",
            )
        return

    ploting_graphs_cont_cat_target_predictors()

    def remove_missing_values(df):
        num_missing = df.isnull().sum()
        print("missing:", num_missing)
        # Remove the rows with missing values
        df = df.dropna()
        # Print the new number of rows and columns in the DataFrame
        # print("New shape of the DataFrame:", df.shape)
        return df

    def diff_mean_plots(df, predictors, class_name):
        df = df.copy()
        for predictor in predictors:
            x = predictor + "_bin_"
            y = "_is_" + predictor.lower()
            df[y] = (
                df[class_name].astype(str).str.lower() == predictor.lower()
            ).astype(int)
            if df[predictor].dtype == "object":
                df[x] = df[predictor]
            else:
                df[x] = pd.cut(df[predictor], bins=10, right=True)

                mid_points = []
                for interval in df[x]:
                    mid_points.append(interval.mid)
                df[x] = mid_points

            df[x + "_midpoint"] = df[x].mean()
            df_for_histogram = df[x].value_counts().to_frame().reset_index()
            mean_bins = df[[x, y]].groupby(x).mean().reset_index()

            # Calculate weighted means
            counts = df[x].value_counts().to_frame().reset_index()
            counts.columns = [x + "_midpoint", "count"]
            counts[x + "_midpoint"] = counts[x + "_midpoint"].astype(float)
            counts = counts.merge(df[[x + "_midpoint", y]], on=x + "_midpoint")

            weighted_means = (
                counts.groupby(x + "_midpoint")[y]
                .apply(lambda x: (x * counts["count"]).sum() / counts["count"].sum())
                .reset_index()
            )
            weighted_means.columns = [x + "_midpoint", y]
            #
            bar_figure = make_subplots(specs=[[{"secondary_y": True}]])

            bar_figure.add_trace(
                go.Bar(
                    x=df_for_histogram["index"], y=df_for_histogram[x], name=predictor
                ),
                secondary_y=False,
            )

            # Add unweighted mean line
            bar_figure.add_trace(
                go.Scatter(
                    x=mean_bins[x],
                    y=mean_bins[y],
                    name="Unweighted",
                    mode="lines + markers",
                    marker=dict(color="crimson"),
                ),
                secondary_y=True,
            )

            # Add weighted mean line
            bar_figure.add_trace(
                go.Scatter(
                    x=weighted_means[x + "_midpoint"],
                    y=weighted_means[y],
                    name="Weighted",
                    mode="lines + markers",
                    marker=dict(color="blue"),
                ),
                secondary_y=True,
            )

            bar_figure.add_trace(
                go.Scatter(
                    x=mean_bins[x],
                    y=mean_bins[y],
                    name=predictor + "bins mean",
                    mode="lines + markers",
                    marker=dict(color="crimson"),
                ),
                secondary_y=True,
            )

            # overall avg of graph
            bar_figure.add_trace(
                go.Scatter(
                    x=mean_bins[x],
                    y=[df[y].mean()] * len(mean_bins[x]),
                    name=predictor + "mean",
                    mode="lines",
                ),
                secondary_y=True,
            )

            bar_figure.update_layout(
                title="Mean Response Plot of " + predictor,
                yaxis=dict(title=dict(text="Response(Target)"), side="left"),
                yaxis2=dict(
                    title=dict(text="Response side"), side="right", overlaying="y"
                ),
                xaxis=dict(title=dict(text=predictor + " Bins")),
            )
            # bar_figure.show()
            bar_figure.write_html(
                file=f"Diff_mean_response/"
                f"diff_mean_response{predictor}_{class_name}.html",
                include_plotlyjs="cdn",
            )

    diff_mean_plots(df, predictors_df, "target")

    def random_forest_model():
        boston_df = df.copy()
        # print(boston_df.columns)
        boston_df["CHAS"] = boston_df["CHAS"].astype(float)

        X = boston_df.drop("target", axis=1)
        y = boston_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        #     # Check for missing values
        #     if X_train.isnull().sum().sum() > 0:
        #         print("Missing values found in the dataset")
        #         return
        # #
        #     # Check for large or infinite values
        #     if not np.isfinite(X_train.values).all():
        #         print("Large or infinite values found in the dataset")
        #         return
        # building a model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Fit the model to the training data
        rf_model.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = rf_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        importances = rf_model.feature_importances_

        # Get the feature names from the dataset
        feature_names = X.columns
        # print("columns:",feature_names)
        # print("importances_col",importances)
        # print(feature_names.shape, importances.shape)

        # Create a dataframe with feature importances
        feature_importances = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        # print(feature_importances)

        # Filter the dataframe to only include continuous predictors
        continuous_features = [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
        ]
        continuous_importances = feature_importances[
            feature_importances["feature"].isin(continuous_features)
        ]

        # Sort the dataframe by importance
        continuous_importances = continuous_importances.sort_values(
            by="importance", ascending=False
        )

        # Print the variable importance rankings
        # print(continuous_importances)
        # Create ranking table
        table = tabulate(
            continuous_importances, headers="keys", tablefmt="github", showindex=False
        )
        # Print the ranking table
        print(table)

        # for html webpage
        table_html = tabulate(
            continuous_importances, headers="keys", tablefmt="html", showindex=False
        )
        with open("output.html", "w") as f:
            f.write(table_html)

    random_forest_model()


if __name__ == "__main__":
    stats = statistics()  # create an instance of the statistics class
    stats.Linear_reg_p_t_stats()  # call the method on the instance
    stats.logit_p_t_stats()
    sys.exit(main())
