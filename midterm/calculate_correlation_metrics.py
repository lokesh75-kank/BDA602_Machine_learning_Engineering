import numpy as np
import pandas as pd
import plotly.graph_objs as go


def calculating_correlation_metrics(df, predictors, response):

    # creating two separate dataframes of predictors and response
    # creating dataframe of all the columns with names from the "predictors" list
    df_pred = df[predictors]
    # print("predictors df:", df_pred.head())

    # creating dataframe of the columns with names from the response
    # df_res = df[response]
    # print("response df:", df_res.head())

    # creating two separate dataframes according to the datatypes of the columns of the predictor dataframe
    categorical_cols = df_pred.select_dtypes(include=["object"]).columns.tolist()
    df_pred_cat = df[categorical_cols]
    # select only columns of dtype 'int' or 'float' (i.e. continuous variables)
    continuos_cols = df_pred.select_dtypes(include=["int", "float"]).columns.tolist()
    df_pred_con = df[continuos_cols]

    # calculate correlation matrix for continuous/continuous pairs
    corr_con_con = df_pred_con.corr()

    # calculate correlation matrix for continuous/categorical pairs
    corr_con_cat = pd.DataFrame(columns=df_pred_con.columns, index=df_pred_cat.columns)
    for col in df_pred_cat.columns:
        for cat_col in df_pred_con.columns:
            corr = df_pred_cat[col].corr(df_pred_con[cat_col], method="spearman")
            corr_con_cat.at[col, cat_col] = corr

    # calculate correlation matrix for categorical/categorical pairs
    corr_cat_cat = pd.DataFrame(
        np.nan, index=df_pred_cat.columns, columns=df_pred_cat.columns
    )
    for col1 in df_pred_cat.columns:
        for col2 in df_pred_cat.columns:
            if col1 == col2:
                corr_cat_cat.at[col1, col2] = 1.0
            else:
                corr_cat_cat.at[col1, col2] = (
                    df_pred_cat[col1] == df_pred_cat[col2]
                ).mean()

    # print all correlation matrices
    # print('\nCategorical / Categorical correlation matrix:')
    # print(corr_cat)
    # print('\nContinuous / Categorical correlation matrix:')
    # print(corr_ccat)

    # print('Continuous / Continuous correlation matrix:')
    # print(corr_cc)

    return corr_cat_cat, corr_con_cat, corr_con_con


# check the datatypes (Categorical / Continuous)
# def check_column_type(dataframe, column_name):
#     column_type = dataframe[column_name].dtypes
#     if column_type == 'object':
#         return f"{column_name} is a categorical column"
#     else:
#         return f"{column_name} is a continuous column"

# printing the result
# for i in range(0,len(predictors)):
#     print(check_column_type(df_pred, predictors[i]))


# Convert correlation matrices to the desired table format.
def convert_to_desired_format(corr_cat_cat, corr_con_cat, corr_con_con):

    # Converting the 2d dataframes to the desired format table
    # Convert corr_cc dataframe to desired format
    corr_con_con = corr_con_con.stack().reset_index()
    corr_con_con.columns = ["cont1", "cont2", "corr"]

    # Convert corr_ccat dataframe to desired format
    corr_con_cat = corr_con_cat.stack().reset_index()
    corr_con_cat.columns = ["cat", "cont", "corr"]

    # Convert corr_cat dataframe to desired format
    corr_cat_cat = corr_cat_cat.stack().reset_index()
    corr_cat_cat.columns = ["cat1", "cat2", "corr"]

    # Putting values in tables in DESC order by correlation metric values
    corr_con_con = corr_con_con.sort_values("corr", ascending=False)

    corr_con_cat = corr_con_cat.sort_values("corr", ascending=False)

    corr_cat_cat = corr_cat_cat.sort_values("corr", ascending=False)
    # corr_cat_df = corr_cat_df.iloc[:, 1:]

    # drop the columns whose correlation metric values are '1.0'
    corr_con_con = corr_con_con[corr_con_con["corr"] != 1.0]
    corr_con_cat = corr_con_cat[corr_con_cat["corr"] != 1.0]
    corr_cat_cat = corr_cat_cat[corr_cat_cat["corr"] != 1.0]

    return corr_cat_cat, corr_con_cat, corr_con_con


def create_corrheatmapfigs(corr_cat_cat, corr_con_cat, corr_con_con):

    # Convert the three matrices into float
    corr_con_con = corr_con_con.astype(float)
    corr_con_cat = corr_con_cat.astype(float)
    corr_cat_cat = corr_cat_cat.astype(float)

    # Continuous / Continuous pairs heatmap
    fig_con_con = go.Figure(
        data=go.Heatmap(
            z=corr_con_con.astype(float).values,
            x=corr_con_con.columns.tolist(),
            y=corr_con_con.columns.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )

    # Continuous / Categorical pairs heatmap

    fig_con_cat = go.Figure(
        data=go.Heatmap(
            z=corr_con_cat.astype(float).values,
            x=corr_con_con.columns.tolist(),
            y=corr_cat_cat.columns.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )

    # Categorical / Categorical pairs heatmap
    fig_cat_cat = go.Figure(
        data=go.Heatmap(
            z=corr_cat_cat.astype(float).values,
            x=corr_cat_cat.columns.tolist(),
            y=corr_cat_cat.columns.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )

    # fig_cc, fig_ccat, and fig_cat are plotly figures
    # create an empty list to store the figures
    corr_figs = []

    # append the figures to the list
    corr_figs.append(fig_cat_cat)
    corr_figs.append(fig_con_cat)
    corr_figs.append(fig_con_con)

    return corr_figs
