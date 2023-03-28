# import statistics
# import sys

# import datasets
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from sklearn.metrics import confusion_matrix

from Assignment_04 import datasets

# from plotly.subplots import make_subplots
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import confusion_matrix, mean_squared_error


# from sklearn.model_selection import train_test_split
# from tabulate import tabulate


datasets = datasets.TestDatasets()
df, predictors, response = datasets.get_test_data_set("diabetes")
# print(df.head())
# print(predictors)


def is_continuous(data, col):
    # This Function takes in a column of a pandas data frame and returns a boolean depending on if the column variables
    # are continuous or not.

    if len(data[col].unique()) <= 3 or data[col].dtype == "O":
        return False
    return True


def mean_diff_tbl(pred, resp, data):

    df = pd.DataFrame(
        columns=[
            "(i)",
            "Bin_Pop",
            "Bin_Mean",
            "Pop_Mean",
            "Pop_Proportion",
            "Mean_Sq_Diff",
            "Mean_Sq_Diff_Weighted",
        ]
    )
    pop_mean = data[resp].mean()
    if not is_continuous(data, pred):
        for i, val in enumerate(data[pred].unique()):
            bin_mean = data[resp][data[pred] == val].mean()
            bin_pop = len(data[resp][data[pred] == val])
            pop_prop = bin_pop / len(data[resp])
            mean_sq_diff = (pop_mean - bin_mean) ** 2
            mean_sq_diff_w = mean_sq_diff * pop_prop
            df.loc[i] = [
                val,
                bin_pop,
                bin_mean,
                pop_mean,
                pop_prop,
                mean_sq_diff,
                mean_sq_diff_w,
            ]
    else:
        data_sorted = data.sort_values(pred)
        data_sorted["bins"] = pd.cut(data_sorted[pred], 10)
        for i, cur_bin in enumerate(data_sorted["bins"].unique()):
            bin_mean = data_sorted[resp][data_sorted["bins"] == cur_bin].mean()
            bin_pop = len(data_sorted[pred][data_sorted["bins"] == cur_bin])
            pop_prop = bin_pop / len(data_sorted[pred])
            mean_sq_diff = (pop_mean - bin_mean) ** 2
            mean_sq_diff_w = mean_sq_diff * pop_prop
            df.loc[i] = [
                i,
                bin_pop,
                bin_mean,
                pop_mean,
                pop_prop,
                mean_sq_diff,
                mean_sq_diff_w,
            ]
    print(df)
    return df


pred = "age"
res = "target"
# wt = unweighted_table(pred,'bmi',res)
newdf = mean_diff_tbl(pred, "target", df)


def plot_mean_diff(df):

    fig = px.bar(x=df["(i)"], y=df["Bin_Mean"])
    fig.add_hline(y=df["Pop_Mean"][0])
    fig.show()


# print(newdf.columns)
# def main():
#     creating object of testdatasets class
#     datasets = TestDatasets()
#     checking other datasets and selecting one
#     df, predictors, response = datasets.get_test_data_set("diabetes")
#     df, predictors, response = datasets.get_test_data_set("breast_cancer")
#     df, predictors, response = datasets.get_test_data_set("boston")
#
#     df["RAD"] = df["RAD"].astype("float64")
#     """
#     note: I dont know if am data manupulation is right way but I noticed
#     that in reference graphs from slide that column type for
#     "Rad" was continuous, also to train regression models
#     data type "object" is not accepted hence converting it to "float".
#     """
#     # print(df.dtypes)
#     predictors_df = df[predictors]
#     # print(predictors_df)
#     # print("col_dtype:",df.columns.dtype)
#
#     def to_check_cat_con_pred(columns):
#         dict = {}
#         for col in columns:
#             # print(col)
#             if columns[col].dtype == "object":
#                 dict[col] = "== Categorical"
#             elif columns[col].dtype in ["int64", "float64"]:
#                 dict[col] = "== Continuous"
#         return dict
#
#     print("checking_predictors_type:", to_check_cat_con_pred(predictors_df))
#
#     def to_check_cat_con_response(column):
#         for i in column:
#             if isinstance(i, float):
#                 print("Continuous")
#             else:
#                 print("Categorical")
#
#     print("checking_response_type:", to_check_cat_con_response(df["target"]))
#
#     # def ploting_graphs_cont_cat_target_predictors():
#     #     for i in predictors_df:
#     #         # print(i)
#     #         x = i
#     #         y = df["target"]
#     #         fig = px.scatter(df, x=x, y=y, trendline="ols")
#     #         if df["CHAS"].dtype == "object":
#     #             fig.update_layout(
#     #                 title="Continuous Response by Catagorical Predictor",
#     #                 xaxis_title=f"Predictor_{i}",
#     #                 yaxis_title=f"Response_{y}",
#     #             )
#     #         else:
#     #             fig.update_layout(
#     #                 title="Continuous Response by Continuous Predictor",
#     #                 xaxis_title=f"Predictor_{i}",
#     #                 yaxis_title=f"Response_{y}",
#     #             )
#     #         # fig.show()
#     #         fig.write_html(
#     #             file=f"plots_cont_cat/"
#     #             f"lecture_6_cont_response_cont_predictor_"
#     #             f"{i}_scatter_plot.html",
#     #             include_plotlyjs="cdn",
#     #         )
#     #     return
#
#     # ploting_graphs_cont_cat_target_predictors()
#
#     def remove_missing_values(df):
#         num_missing = df.isnull().sum()
#         print("missing:", num_missing)
#         # Remove the rows with missing values
#         df = df.dropna()
#         # Print the new number of rows and columns in the DataFrame
#         # print("New shape of the DataFrame:", df.shape)
#         return df
#
#     remove_missing_values(df)
#
#     diff_mean_plots(df, predictors_df, "target")

# def random_forest_model():
#     boston_df = df.copy()
#     # print(boston_df.columns)
#     boston_df["CHAS"] = boston_df["CHAS"].astype(float)
#
#     X = boston_df.drop("target", axis=1)
#     y = boston_df["target"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#
#     #     # Check for missing values
#     #     if X_train.isnull().sum().sum() > 0:
#     #         print("Missing values found in the dataset")
#     #         return
#     # #
#     #     # Check for large or infinite values
#     #     if not np.isfinite(X_train.values).all():
#     #         print("Large or infinite values found in the dataset")
#     #         return
#     # building a model
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     # Fit the model to the training data
#     rf_model.fit(X_train, y_train)
#     # Make predictions on the test data
#     y_pred = rf_model.predict(X_test)
#
#     mse = mean_squared_error(y_test, y_pred)
#     print("Mean Squared Error:", mse)
#
#     importances = rf_model.feature_importances_
#
#     # Get the feature names from the dataset
#     feature_names = X.columns
#     # print("columns:",feature_names)
#     # print("importances_col",importances)
#     # print(feature_names.shape, importances.shape)
#
#     # Create a dataframe with feature importances
#     feature_importances = pd.DataFrame(
#         {"feature": feature_names, "importance": importances}
#     )
#     # print(feature_importances)
#
#     # Filter the dataframe to only include continuous predictors
#     continuous_features = [
#         "CRIM",
#         "ZN",
#         "INDUS",
#         "CHAS",
#         "NOX",
#         "RM",
#         "AGE",
#         "DIS",
#         "RAD",
#         "TAX",
#         "PTRATIO",
#         "B",
#         "LSTAT",
#     ]
#     continuous_importances = feature_importances[
#         feature_importances["feature"].isin(continuous_features)
#     ]
#
#     # Sort the dataframe by importance
#     continuous_importances = continuous_importances.sort_values(
#         by="importance", ascending=False
#     )
#
#     # Print the variable importance rankings
#     # print(continuous_importances)
#     # Create ranking table
#     table = tabulate(
#         continuous_importances, headers="keys", tablefmt="github", showindex=False
#     )
#     # Print the ranking table
#     print(table)
#
#     # for html webpage
#     table_html = tabulate(
#         continuous_importances, headers="keys", tablefmt="html", showindex=False
#     )
#     with open("output.html", "w") as f:
#         f.write(table_html)

# random_forest_model()


def plot_catp_catr(pred, resp, data):
    # This function plots a heatmap of the two catigrocial variables
    data[pred] = data[pred].astype("string")
    conf_matrix = confusion_matrix(data[pred], data[resp])
    fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))
    fig.update_layout(
        title=f"Categorical Predictor: {pred} by Categorical Response {resp}",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig.show()
    # fig.write_html(
    #     file=f"./plots/cat_{resp}_by_cat_{pred}_heatmap.html",
    #     include_plotlyjs="cdn",
    # )


def plot_catp_contr(pred, resp, data):
    data[pred] = data[pred].astype(
        "string"
    )  # made this a string since its a catigorical variable
    cat_pred_list = list(data[pred].unique())
    cont_resp_list = [
        data[resp][data[pred] == pred_name] for pred_name in cat_pred_list
    ]

    fig_1 = ff.create_distplot(cont_resp_list, cat_pred_list, bin_size=0.2)
    fig_1.update_layout(
        title=f"Continuous Response: {resp} by Categorical Predictor: {pred}",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    fig_1.show()
    # fig_1.write_html(
    #     file=f"./plots/cont_{resp}_cat_{pred}_dist_plot.html",
    #     include_plotlyjs="cdn",
    # )


def plot_contp_catr(pred, resp, data):
    data[resp] = data[resp].astype("string")
    cat_resp_list = list(data[resp].unique())
    n = len(cat_resp_list)
    cont_pred_list = [
        data[pred][data[resp] == resp_name] for resp_name in cat_resp_list
    ]
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(cont_pred_list, cat_resp_list):
        fig_2.add_trace(
            go.Violin(
                x=np.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title=f"Continuous Predictor: {pred} by Categorical Response: {resp}",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_2.show()
    # fig_2.write_html(
    #     file=f"./plots/cat_{resp}_cont_{pred}_violin_plot.html",
    #     include_plotlyjs="cdn",
    # )


def plot_contp_contr(pred, resp, data):
    fig = px.scatter(x=data[pred], y=data[resp], trendline="ols")
    fig.update_layout(
        title=f"Continuous Response: {resp} by Continuous Predictor: {pred}",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()
    # fig.write_html(
    #     file=f"./plots/cont_{resp}_cont_{pred}_scatter_plot.html",
    #     include_plotlyjs="cdn",
    # )


def calc_reg(pred, resp, data):
    predictor = sm.add_constant(data[pred])
    model = sm.OLS(data[resp], predictor)
    model_fit = model.fit()
    p_val = "{:.6e}".format(model_fit.pvalues[1])
    t_val = round(model_fit.tvalues[1], 6)

    print(f"Predictor: {pred}")
    print(model_fit.summary())

    fig = px.scatter(x=data[pred], y=data[resp], trendline="ols")
    fig.update_layout(
        title=f"Variable: {pred}: (t-value={t_val}) (p-value={p_val})",
        xaxis_title=f"Variable: {pred}",
        yaxis_title="y",
    )
    fig.show()

    # fig.write_html(
    #     file=f"./plots/cont_{resp}_cont_{pred}_reg_plot.html",
    #     include_plotlyjs="cdn",
    # )

    return p_val, t_val


def calc_log_reg(pred, resp, data):
    model_fit = sm.Logit(data[resp], data[pred]).fit()
    print(model_fit.summary())
    p_val = "{:.6e}".format(model_fit.pvalues[0])
    t_val = round(model_fit.tvalues[0], 6)
    fig = px.scatter(x=data[pred], y=data[resp], trendline="ols")
    fig.update_layout(
        title=f"Variable: {pred}: (t-value={t_val}) (p-value={p_val})",
        xaxis_title=f"Variable: {pred}",
        yaxis_title="y",
    )
    fig.show()

    # fig.write_html(
    #     file=f"./plots/cont_{resp}_cont_{pred}_log_reg_plot.html",
    #     include_plotlyjs="cdn",
    # )
    return p_val, t_val


def main():
    variable_df = pd.DataFrame(
        columns=[
            "Predictor",
            "Response",
            "P Value",
            "T Value",
            "Regression Plot",
            "General Plot",
            "Difference of Mean Plot",
            "Mean Squared Differnece Weighted",
        ]
    )

    data_df, predictors, response = datasets.get_test_data_set()
    # col = "cyl"
    # plot_catp_contr(col, response, data_df)
    # msd_tbl = mean_diff_tbl(col, response, data_df)
    # sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
    # plot_mean_diff(msd_tbl)
    # print(col,response)
    # p_val, t_val = calc_reg(col, response, data_df)

    if is_continuous(data_df, response):
        for i, col in enumerate(
            data_df.loc[:, ~data_df.columns.isin([response, "Unnamed: 0"])].columns
        ):
            if is_continuous(data_df, col):
                plot_contp_contr(col, response, data_df)
                msd_tbl = mean_diff_tbl(col, response, data_df)
                sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
                plot_mean_diff(msd_tbl)
                p_val, t_val = calc_reg(col, response, data_df)
            else:
                plot_catp_contr(col, response, data_df)
                msd_tbl = mean_diff_tbl(col, response, data_df)
                sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
                plot_mean_diff(msd_tbl)
                p_val, t_val = calc_reg(col, response, data_df)
            variable_df.loc[i] = [
                col,
                response,
                p_val,
                t_val,
                "see output",
                "see output",
                "see output",
                sum_msdw,
            ]
    else:
        for i, col in enumerate(
            data_df.loc[:, ~data_df.columns.isin([response, "Unnamed: 0"])].columns
        ):

            if is_continuous(data_df, col):
                plot_contp_catr(col, response, data_df)
                msd_tbl = mean_diff_tbl(col, response, data_df)
                sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
                plot_mean_diff(msd_tbl)

            else:
                plot_catp_catr(col, response, data_df)
                msd_tbl = mean_diff_tbl(col, response, data_df)
                sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
                plot_mean_diff(msd_tbl)

            variable_df.loc[i] = [
                col,
                response,
                "Catigorical Response",
                "Catigorical Response",
                "see output",
                "see output",
                "see output",
                sum_msdw,
            ]
    return variable_df


# if __name__ == "__main__":
#     # stats = statistics()  # create an instance of the statistics class
#     # stats.Linear_reg_p_t_stats()  # call the method on the instance
#     # stats.logit_p_t_stats()
#     sys.exit(main())


# df = df.copy()
# # for predictor in predictors:
# x = predictor + "_bin_"
# y = "_is_" + predictor.lower()
# df[y] = (
#         df[class_name].astype(str).str.lower() == predictor.lower()
# ).astype(int)
# if df[predictor].dtype == "object":
#     df[x] = df[predictor]
# else:
#     df[x] = pd.cut(df[predictor], bins=10, right=True)
#
#     mid_points = []
#     for interval in df[x]:
#         mid_points.append(interval.mid)
#     df[x] = mid_points
#
# df[x + "_midpoint"] = df[x].mean()
# df_for_histogram = df[x].value_counts().to_frame().reset_index()
# mean_bins = df[[x, y]].groupby(x).mean().reset_index()
#
# # Calculate weighted means
# counts = df[x].value_counts().to_frame().reset_index()
# counts.columns = [x + "_midpoint", "count"]
# counts[x + "_midpoint"] = counts[x + "_midpoint"].astype(float)
# counts = counts.merge(df[[x + "_midpoint", y]], on=x + "_midpoint")
#
# weighted_means = (
#     counts.groupby(x + "_midpoint")[y]
#     .apply(lambda x: (x * counts["count"]).sum() / counts["count"].sum())
#     .reset_index()
# )
# print("weighted mean:", weighted_means)
# weighted_means.columns = [x + "_midpoint", y]
# #
# bar_figure = make_subplots(specs=[[{"secondary_y": True}]])
#
# bar_figure.add_trace(
#     go.Bar(
#         x=df_for_histogram["index"], y=df_for_histogram[x], name=predictor
#     ),
#     secondary_y=False,
# )
# # Add unweighted mean line
# bar_figure.add_trace(
#     go.Scatter(
#         x=mean_bins[x],
#         y=mean_bins[y],
#         name="Unweighted",
#         mode="lines + markers",
#         marker=dict(color="crimson"),
#     ),
#     secondary_y=True,
# )
#
# # Add weighted mean line
# bar_figure.add_trace(
#     go.Scatter(
#         x=weighted_means[x + "_midpoint"],
#         y=weighted_means[y],
#         name="Weighted",
#         mode="lines + markers",
#         marker=dict(color="blue"),
#     ),
#     secondary_y=True,
# )
#
# bar_figure.add_trace(
#     go.Scatter(
#         x=mean_bins[x],
#         y=mean_bins[y],
#         name=predictor + "bins mean",
#         mode="lines + markers",
#         marker=dict(color="crimson"),
#     ),
#     secondary_y=True,
# )
#
# # overall avg of graph
# bar_figure.add_trace(
#     go.Scatter(
#         x=mean_bins[x],
#         y=[df[y].mean()] * len(mean_bins[x]),
#         name=predictor + "mean",
#         mode="lines",
#     ),
#     secondary_y=True,
# )
#
# bar_figure.update_layout(
#     title="Mean Response Plot of " + predictor,
#     yaxis=dict(title=dict(text="Response(Target)"), side="left"),
#     yaxis2=dict(
#         title=dict(text="Response side"), side="right", overlaying="y"
#     ),
#     xaxis=dict(title=dict(text=predictor + " Bins")),
# )
# bar_figure.show()
# bar_figure.write_html(
#     file=f"Diff_mean_response/"
#     f"diff_mean_response{predictor}_{class_name}.html",
#     include_plotlyjs="cdn",
# )

# predic1 = predictors_df['CRIM']
