import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

continuous_pred = []
catagorical_pred = []


def predictor_plots(df_pred, df_res):

    my_fig_con = []
    my_fig_cat = []

    if df_res.dtype == "object":
        # if df_res.dtype == "object" or df_res.nunique() <= 2:
        # Categorical Response
        # Check if predictor is Boolean / Categorical
        for col in df_pred.columns:
            if (
                df_pred[col].dtype == "object"
                or df_pred[col].dtype.name == "category"
                # or df_pred[col].nunique() <= 2
            ):
                catagorical_pred.append(col)

                fig = px.histogram(df_pred, x=col, color=df_res)
                fig.update_layout(title=f"Histogram of {col} Or by Response")
                my_fig_cat.append(fig)
                # fig.show()
            else:
                continuous_pred.append(col)

                fig = px.violin(df_pred, y=col, x=df_res, box=True, points="all")
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
            if df_pred[col].dtype == "object" or df_pred[col].nunique() <= 2:
                catagorical_pred.append(col)
                print("cat_list", catagorical_pred)

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
                # fig = px.histogram(df_pred, x=col, color=col, barmode="overlay")
                # fig.update_layout(title=f"Distribution of Response by {col}")
                my_fig_cat.append(fig)
                # fig.show()
            else:
                continuous_pred.append(col)

                fig = px.scatter(df_pred, x=col, y=df_res, trendline="ols")
                fig.update_layout(title="Scatter Plot of Response vs Weight")
                my_fig_con.append(fig)
                # fig.show()

    # CREATE HTML FILES OF PLOTS

    # Create directory if it doesn't exist
    if not os.path.exists("Categorical_Predictors_Plots"):
        os.makedirs("Categorical_Predictors_Plots")

    if not os.path.exists("Continuous_Predictors_Plots"):
        os.makedirs("Continuous_Predictors_Plots")

    # Save each figure to HTML file in the directory

    for i, fig in enumerate(my_fig_cat):
        file_name = f"plot_{i}.html"
        file_path = os.path.join("Categorical_Predictors_Plots", file_name)
        pio.write_html(fig, file_path, include_plotlyjs="cdn")

    for i, fig in enumerate(my_fig_con):
        file_name = f"plot_{i}.html"
        file_path = os.path.join("Continuous_Predictors_Plots", file_name)
        pio.write_html(fig, file_path, include_plotlyjs="cdn")

    return continuous_pred, catagorical_pred


def create_predictor_dfs(df, response, continuous_pred, catagorical_pred):

    my_fig_con = []
    my_fig_cat = []

    con_col2 = []
    con_col3 = []

    for i in range(0, len(continuous_pred)):
        link = "'Continuous_Predictors_Plots/plot_" + str(i) + ".html'"
        temp = "<a href=" + link + ">" + continuous_pred[i] + "</a>"
        con_col2.append(temp)

    for i in range(0, len(continuous_pred)):
        link = (
            "'Continuous_Mean_of_Response_Plot/Mean_of_Response_Plot_"
            + continuous_pred[i]
            + ".html'"
        )
        temp = "<a href=" + link + ">" + continuous_pred[i] + "</a>"
        con_col3.append(temp)

    continuous_pred_df = pd.DataFrame(
        {
            "Feature": continuous_pred,
            "Plot": con_col2,
            "Mean of Response Plot": con_col3,
            "Diff Mean Response (Weighted)": [np.nan] * len(continuous_pred),
            "Diff Mean Response (Unweighted)": [np.nan] * len(continuous_pred),
            "P-Value": [np.nan] * len(continuous_pred),
            "T-Score": [np.nan] * len(continuous_pred),
        }
    )

    cat_col2 = []
    cat_col3 = []

    for i in range(0, len(catagorical_pred)):
        link = "'Categorical_Predictors_Plots/plot_" + str(i) + ".html'"
        temp = "<a href=" + link + 'target="_blank">' + catagorical_pred[i] + "</a>"
        cat_col2.append(temp)

    for i in range(0, len(catagorical_pred)):
        link = (
            "'Categorical_Mean_of_Response_Plot/Mean_of_Response_Plot_"
            + catagorical_pred[i]
            + ".html'"
        )
        # link = "'Categorical_Mean_of_Response_Plot/plot_"+str(i)+".html'"
        temp = "<a href=" + link + 'target="_blank">' + catagorical_pred[i] + "</a>"
        cat_col3.append(temp)

    catagorical_pred_df = pd.DataFrame(
        {
            "Feature": catagorical_pred,
            "Plot": cat_col2,
            "Mean of Response Plot": cat_col3,
            "Diff Mean Response (Weighted)": [np.nan] * len(catagorical_pred),
            "Diff Mean Response (Unweighted)": [np.nan] * len(catagorical_pred),
        }
    )

    for i, pred in enumerate(continuous_pred):
        uw_mse, w_mse, fig = weighted_unweighted_table_con(df, pred, response)
        p, t = p_t_values(df, pred, response)
        continuous_pred_df.loc[i, "Diff Mean Response (Weighted)"] = w_mse
        continuous_pred_df.loc[i, "Diff Mean Response (Unweighted)"] = uw_mse
        continuous_pred_df.loc[i, "P-Value"] = p
        continuous_pred_df.loc[i, "T-Score"] = t
        my_fig_con.append(fig)

    for i, pred in enumerate(catagorical_pred):
        uw_mse, w_mse, fig = weighted_unweighted_table_cat(df, pred, response)
        catagorical_pred_df.loc[i, "Diff Mean Response (Weighted)"] = w_mse
        catagorical_pred_df.loc[i, "Diff Mean Response (Unweighted)"] = uw_mse
        my_fig_cat.append(fig)

    # CREATE HTML FILES OF Mean of Response Plots

    # Create directory if it doesn't exist
    if not os.path.exists("Categorical_Mean_of_Response_Plot"):
        os.makedirs("Categorical_Mean_of_Response_Plot")

    if not os.path.exists("Continuous_Mean_of_Response_Plot"):
        os.makedirs("Continuous_Mean_of_Response_Plot")

    # Save each figure to HTML file in the directory

    for i, fig in enumerate(my_fig_cat):
        var = catagorical_pred[i]
        file_name = f"Mean_of_Response_Plot_{var}.html"
        file_path = os.path.join("Categorical_Mean_of_Response_Plot", file_name)
        pio.write_html(fig, file_path, include_plotlyjs="cdn")

    for i, fig in enumerate(my_fig_con):
        var = continuous_pred[i]
        file_name = f"Mean_of_Response_Plot_{var}.html"
        file_path = os.path.join("Continuous_Mean_of_Response_Plot", file_name)
        pio.write_html(fig, file_path, include_plotlyjs="cdn")

    return continuous_pred_df, catagorical_pred_df


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

    return uw_mse, w_mse, fig


def p_t_values(df, continuous_pred, response):
    y = df[response]
    pred = df[continuous_pred]
    predictor = sm.add_constant(pred)
    linear_regression_model = sm.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    # print(f"Variable: {continuous_pred}")
    # print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[0], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[0])

    # print("pvalues:", p_value)
    # print("tvalues:", t_value)
    return p_value, t_value


# def impurity_based_feature_importance(df, continuous_pred_df, response):
#     numeric_columns = continuous_pred_df.select_dtypes(include=['float', 'int']).columns
#     X = continuous_pred_df[numeric_columns]
#     y = df[response]
#
#     model = RandomForestRegressor(n_estimators=100, random_state=None)
#     model.fit(X, y)
#     # Compute impurity based feature importance
#     feature_importance = model.feature_importances_
#     print("feature Importance:", feature_importance)
#
#     return feature_importance


def impurity_based_feature_importance(df, response):
    # numeric_columns = continuous_pred_df.select_dtypes(include=['float', 'int']).columns
    X = df.drop("home_team_wins", axis=1)
    y = df[response]

    model = RandomForestRegressor(n_estimators=100, random_state=None)
    model.fit(X, y)
    # Compute impurity based feature importance
    feature_importance = model.feature_importances_
    # Get column names
    column_names = X.columns

    # Print column names and their feature importances
    print("Feature Importance:")
    for name, importance in zip(column_names, feature_importance):
        print(f"{name}: {importance}")

    return feature_importance


def weighted_unweighted_table_cat(data, feature_name, response_name):
    feature = data[feature_name]
    Y = data[response_name]

    categories = feature.unique()
    table = pd.DataFrame(columns=["Category", "CategoryCount", "CategoryMean"])

    for cat in categories:
        cat_mean = data[feature == cat][response_name].mean()
        cat_count = data[feature == cat][response_name].count()

        new_table = {
            "Category": cat,
            "CategoryCount": cat_count,
            "CategoryMean": cat_mean,
        }

        table = table.append(new_table, ignore_index=True)

    pop_mean = Y.mean()

    table["PopulationMean"] = pop_mean
    table["MeanSquareDiff"] = [((i - pop_mean) ** 2) for i in table["CategoryMean"]]
    table["PopulationProportion"] = [
        i / table["CategoryCount"].sum() for i in table["CategoryCount"]
    ]
    table["WeightedMSD"] = table["MeanSquareDiff"] * table["PopulationProportion"]

    uw_mse = table["MeanSquareDiff"].sum() / len(categories)
    w_mse = table["WeightedMSD"].sum()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=table["Category"],
            y=table["CategoryCount"],
            name="Population",
            yaxis="y2",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["Category"],
            y=table["CategoryMean"],
            name="Category Mean(μ𝑖)",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["Category"],
            y=table["PopulationMean"],
            name="Population Mean(μpop)",
            yaxis="y",
            mode="lines",
        )
    )
    fig.update_layout(
        xaxis_title="Predictor Category",
        yaxis_title="Response",
        yaxis2={"title": "Population", "side": "right", "overlaying": "y"},
    )

    # fig.show()

    # print("uw_mse, w_mse", uw_mse, w_mse)
    # print("table.columns", table.columns)

    return uw_mse, w_mse, fig
