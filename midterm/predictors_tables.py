import os
import statistics

import numpy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as pyo
import statsmodels
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

from midterm.brute_force import form_pairs

# cat=object
# con=float64, int64, bool

continuous_pred = []
catagorical_pred = []


def predictor_plots(df_pred, df_res):
    # check if response is categorical or continuous

    my_fig_con = []
    my_fig_cat = []

    # print("df_res.dtype",df_res.dtype)
    # print(("df_pred.dtype",df_pred.dtypes))

    if df_res.dtype == "object":
        # Categorical Response
        # Check if predictor is Boolean / Categorical
        for col in df_pred.columns:

            if df_pred[col].dtype == "object" or df_pred[col].dtype.name == "category":
                catagorical_pred.append(col)
                # print("check 1",catagorical_pred)

                fig = px.histogram(df_pred, x=col, color=df_res)
                fig.update_layout(title=f"Histogram of {col} Or by Response")
                my_fig_cat.append(fig)
                # fig.show()
            else:
                continuous_pred.append(col)
                # print("check 2",continuous_pred)

                fig = px.violin(
                    df_pred, y=col, x=df_res, box=True, points="all", color="origin"
                )
                fig.update_layout(title=f"Violin Plot of {col} by Response and Origin")
                # fig.show()
                fig = px.histogram(df_pred, x=col, color=df_res, barmode="overlay")
                fig.update_layout(title=f"Distribution of {col} by Response")
                my_fig_con.append(fig)
                # fig.show()

    elif df_res.dtype == "float64" or df_res.dtype == "int64" or df_res.dtype == "bool":
        # print(df_res.dtype)
        # Continuous Response
        # Check if predictor is Boolean / Categorical
        for col in df_pred.columns:
            if df_pred[col].dtype == "object" or df_pred[col].dtype.name == "category":
                catagorical_pred.append(col)
                # print("check 3",catagorical_pred)

                # df_pred = df_pred.sort_values("origin")
                fig = px.violin(
                    df_res,
                    y=df_res.to_frame().columns[0],
                    x=df_pred[col],
                    color=df_pred[col],
                    box=True,
                )
                fig.update_layout(title=f"Violin Plot of Response by {col}")
                # fig.show()
                fig = px.histogram(df_pred, x=col, color=col, barmode="overlay")
                fig.update_layout(title=f"Distribution of Response by {col}")
                my_fig_cat.append(fig)
                # fig.show()
            else:
                continuous_pred.append(col)
                # print("check 4",continuous_pred)

                fig = px.scatter(df_pred, x=col, y=df_res, trendline="ols")
                fig.update_layout(title="Scatter Plot of Response vs Weight")
                my_fig_con.append(fig)
                # fig.show()

    print("continuous_pred", continuous_pred)
    print("catagorical_pred", catagorical_pred)

    # ////////////////////CREATE HTML FILES OF PLOTS///////////////

    # Create directory if it doesn't exist
    if not os.path.exists("Categorical_Predictors_Plots"):
        os.makedirs("Categorical_Predictors_Plots")

    if not os.path.exists("Continuous_Predictors_Plots"):
        os.makedirs("Continuous_Predictors_Plots")

    # Save each figure to HTML file in the directory

    for i, fig in enumerate(my_fig_cat):
        file_name = f"plot_{i}.html"
        file_path = os.path.join("Categorical_Predictors_Plots", file_name)
        pio.write_html(fig, file_path)

    for i, fig in enumerate(my_fig_con):
        file_name = f"plot_{i}.html"
        file_path = os.path.join("Continuous_Predictors_Plots", file_name)
        pio.write_html(fig, file_path)


################################################################################


def create_predictor_dfs(df, response):

    continuous_pred_df = pd.DataFrame(
        {
            "Feature": continuous_pred,
            "Plot": continuous_pred,
            "Mean of Response Plot": continuous_pred,
            "Diff Mean Response (Weighted)": [np.nan] * len(continuous_pred),
            "Diff Mean Response (Unweighted)": [np.nan] * len(continuous_pred),
        }
    )

    catagorical_pred_df = pd.DataFrame(
        {
            "Feature": catagorical_pred,
            "Plot": catagorical_pred,
            "Mean of Response Plot": catagorical_pred,
        }
    )

    # for pred in continuous_pred:
    #     uw_mse, w_mse = unweighted_table_con(df,pred,response)
    #     continuous_pred_df = continuous_pred_df.append({'uw_mse': uw_mse, 'w_mse': w_mse}, ignore_index=True)

    for i, pred in enumerate(continuous_pred):
        uw_mse, w_mse = unweighted_table_con(df, pred, response)
        continuous_pred_df.loc[i, "Diff Mean Response (Weighted)"] = w_mse
        continuous_pred_df.loc[i, "Diff Mean Response (Unweighted)"] = uw_mse

    # return continuous_pred_df, catagorical_pred_df
    print("continuous_pred_df\n", continuous_pred_df)
    print("catagorical_pred_df\n", catagorical_pred_df)

    return continuous_pred_df, catagorical_pred_df, continuous_pred, catagorical_pred


def weighted_unweighted_table_con(data, feature_name, response_name):
    # print("features name",feature_name)
    feature = data[feature_name]
    Y = data[response_name]

    number_bins = 10
    min_feature = feature.min()
    max_feature = feature.max()
    bin_size = (max_feature - min_feature) / number_bins

    table = pd.DataFrame(
        columns=["LowerBin", "UpperBin", "BinCenter", "BinCount", "BinMean"]
    )

    for n in range(number_bins):
        low = min_feature + (bin_size * n)
        high = min_feature + (bin_size * (n + 1))
        bin_mean = data[(feature >= low) & (feature < high)][response_name].mean()
        bin_count = data[(feature >= low) & (feature < high)][response_name].count()

        if n == 9:
            bin_mean = data[(feature >= low) & (feature <= high)][response_name].mean()
            bin_count = data[(feature >= low) & (feature <= high)][
                response_name
            ].count()

        bin_center = (low + high) / 2

        new_table = {
            "LowerBin": low,
            "UpperBin": high,
            "BinCenter": bin_center,
            "BinCount": bin_count,
            "BinMean": bin_mean,
        }

        table = table.append(new_table, ignore_index=True)

    pop_mean = Y.mean()
    # mean_sq_diff = np.sum((table["BinMean"] - pop_mean) ** 2) / len(table)

    table["PopulationMean"] = pop_mean
    table["MeanSquareDiff"] = [((i - pop_mean) ** 2) for i in table["BinMean"]]
    table["PopulationProportion"] = [
        i / table["BinCount"].sum() for i in table["BinCount"]
    ]
    table["WeightedMSD"] = table["MeanSquareDiff"] * table["PopulationProportion"]

    uw_mse = table["MeanSquareDiff"].sum() / 10
    w_mse = table["WeightedMSD"].sum()

    # fig = px.bar(x=table['BinCenter'], y=table["BinCount"])
    # fig.add_hline(y=table["PopulationMean"][0])
    # fig.add_trace(px.imshow(x=table['BinCenter'], y=table["BinMean"]))
    #
    # fig.update_layout(
    #     xaxis_title=f"Predictors Bin {feature_name}",
    #     yaxis_title="Response"
    # )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=table["BinCenter"],
            y=table["BinCount"],
            name="Population",
            yaxis="y2",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["BinCenter"],
            y=table["BinMean"],
            name="Bin Mean(μ𝑖)",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["BinCenter"],
            y=table["PopulationMean"],
            name="Population Mean(μpop)",
            yaxis="y",
            mode="lines",
        )
    )
    fig.update_layout(
        xaxis_title="Predictor Bin",
        yaxis_title="Response",
        yaxis2={"title": "Population", "side": "right", "overlaying": "y"},
    )

    # fig.show()
    print("uw_mse, w_mse", uw_mse, w_mse)
    print("table.columns", table.columns)

    return uw_mse, w_mse


def unweighted_table_con(data, feature_name, response_name):
    # print(feature_name)
    feature = data[feature_name]
    Y = data[response_name]
    y = Y.to_list()

    table = pd.DataFrame(
        columns=[
            "Category",
            "BinCount",
            "BinMean",
            "PopulationMean",
            "MeanSquareDiff",
        ]
    )
    categories = feature.unique()
    number_bins = len(categories)

    # mean square unweighted table
    for category in categories:
        feature_bin_list = []
        response_bin_list = []
        for i in range(len(feature)):
            if feature[i] == category:
                feature_bin_list.append(feature[i])
                response_bin_list.append(y[i])
        bin_count = len(feature_bin_list)
        bin_mean = statistics.mean(response_bin_list)
        pop_mean = numpy.nanmean(y)
        mean_sq_diff = round(abs((bin_mean - pop_mean) ** 2), 5)
        new_table = {
            "Category": category,
            "BinCount": bin_count,
            "BinMean": bin_mean,
            "PopulationMean": pop_mean,
            "MeanSquareDiff": mean_sq_diff,
        }
        table = table.append(new_table, ignore_index=True)

    fig = px.bar(x=table.index, y=table["BinMean"])
    fig.add_hline(y=table["PopulationMean"][0])
    fig.update_layout(xaxis_title="Predictors Bin", yaxis_title="Response")
    fig.show()

    print("unweighted table:", table)

    return table


import plotly.express as px
import statsmodels.api as sm


def pval_tscore(df, predictors, response):
    # Fit OLS model and extract p-value and t-score for each predictor
    X = df[predictors]
    X = sm.add_constant(X)
    y = df[response]
    model = sm.OLS(y, X).fit()
    pvalues = model.pvalues[1:]
    tscores = model.tvalues[1:]

    # Create DataFrame with p-values and t-scores
    result_df = pd.DataFrame(
        {"Predictor": predictors, "p-value": pvalues, "t-score": tscores}
    )

    # Plot p-values and t-scores using Plotly
    # fig = px.bar(result_df, x='Predictor', y='p-value', title='P-values for Predictors')
    # fig.show()
    #
    # fig = px.bar(result_df, x='Predictor', y='t-score', title='T-scores for Predictors')
    # fig.show()

    print(result_df)


def p_t_values(df, continuous_pred, response):
    print("#####################")
    y = df[response]
    pred = df[continuous_pred]
    predictor = statsmodels.api.add_constant(pred)
    linear_regression_model = statsmodels.api.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {continuous_pred}")
    # print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    print("pvalues:", p_value)
    print("pvalues:", t_value)

    # Plot the figure
    # fig = px.scatter(x=column, y=y, trendline="ols")
    # fig.update_layout(
    #     title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
    #     xaxis_title=f"Variable: {feature_name}",
    #     yaxis_title="y",
    # )
    # fig.show()
    # fig.write_html(
    #     file=f"../../plots/lecture_6_var_{idx}.html", include_plotlyjs="cdn"
    # )
    return p_value, t_value


def impurity_based_feature_importance(df, continuous_pred_df, response):
    X = continuous_pred_df
    y = df[response]

    model = RandomForestRegressor(n_estimators=100, random_state=None)
    model.fit(X, y)
    # Compute impurity based feature importance
    feature_importance = model.feature_importances_
    print("feature 0mp:", feature_importance)

    return feature_importance


# def ranking_con_con(col1, col2, data, feature_name, response_name):
#     # print("features name",feature_name)
#     feature = data[feature_name]
#     Y = data[response_name]
#
#     number_bins = 10
#     min_feature = feature.min()
#     max_feature = feature.max()
#     bin_size = (max_feature - min_feature) / number_bins
#
#     table = pd.DataFrame(
#         columns=["LowerBin", "UpperBin", "BinCenter", "BinCount", "BinMean"]
#     )
#
#     for n in range(number_bins):
#         low = min_feature + (bin_size * n)
#         high = min_feature + (bin_size * (n + 1))
#         bin_mean = data[(feature >= low) & (feature < high)][response_name].mean()
#         bin_count = data[(feature >= low) & (feature < high)][response_name].count()
#
#         if n == 9:
#             bin_mean = data[(feature >= low) & (feature <= high)][response_name].mean()
#             bin_count = data[(feature >= low) & (feature <= high)][response_name].count()
#
#         bin_center = (low + high) / 2
#
#         new_table = {
#             "LowerBin": low,
#             "UpperBin": high,
#             "BinCenter": bin_center,
#             "BinCount": bin_count,
#             "BinMean": bin_mean,
#         }
#
#         table = table.append(new_table, ignore_index=True)
#
#     pop_mean = Y.mean()
#     # mean_sq_diff = np.sum((table["BinMean"] - pop_mean) ** 2) / len(table)
#
#     table["PopulationMean"] = pop_mean
#     table["MeanSquareDiff"] = [((i - pop_mean) ** 2) for i in table["BinMean"]]
#     table["PopulationProportion"] = [i / table["BinCount"].sum() for i in table["BinCount"]]
#     table["WeightedMSD"] = table["MeanSquareDiff"] * table["PopulationProportion"]
#
#     uw_mse = table["MeanSquareDiff"].sum()/10
#     w_mse = table["WeightedMSD"].sum()
#
#     # fig = px.bar(x=table['BinCenter'], y=table["BinCount"])
#     # fig.add_hline(y=table["PopulationMean"][0])
#     # fig.add_trace(px.imshow(x=table['BinCenter'], y=table["BinMean"]))
#     #
#     # fig.update_layout(
#     #     xaxis_title=f"Predictors Bin {feature_name}",
#     #     yaxis_title="Response"
#     # )
#     fig = go.Figure()
#     fig.add_trace(
#         go.Bar(
#             x=table['BinCenter'], y=table["BinCount"], name="Population", yaxis="y2", opacity=0.5
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=table['BinCenter'],
#             y=table["BinMean"],
#             name="Bin Mean(μ𝑖)",
#             yaxis="y",
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=table['BinCenter'],
#             y=table['PopulationMean'],
#             name="Population Mean(μpop)",
#             yaxis="y",
#             mode="lines",
#         )
#     )
#     fig.update_layout(xaxis_title="Predictor Bin",
#                       yaxis_title="Response",
#                       yaxis2 = {"title": "Population","side":"right","overlaying":"y"})
#
#     # fig.show()
#     print("uw_mse, w_mse", uw_mse, w_mse)
#     print("table.columns",table.columns)
#
#     return uw_mse, w_mse
