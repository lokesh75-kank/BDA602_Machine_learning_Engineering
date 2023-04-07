import statistics
import sys

import numpy
import numpy as np
import pandas as pd
import researchpy as rp
import statsmodels
from dataset_loader import TestDatasets
from scipy.stats import f_oneway, pointbiserialr

from midterm.brute_force import (
    brute_force,
    cat_cat_brute_force,
    cat_con_brute_force,
    con_con_brute_force,
)
from midterm.calculate_correlation_metrics import (
    calculating_correlation_metrics,
    convert_to_desired_format,
    create_corrheatmapfigs,
)
from midterm.creating_html import creating_html
from midterm.predictors_tables import (
    create_predictor_dfs,
    impurity_based_feature_importance,
    p_t_values,
    predictor_plots,
)


def main():

    # taking sample dataframe for testing purpose
    test_datasets = TestDatasets()
    df, predictors, response = test_datasets.get_test_data_set("mpg")

    ##########################CODE FOR CORRELATION 3 CORRELATION MATRICES AND 3 PLOTS########################

    corr_cat_cat, corr_con_cat, corr_con_con = calculating_correlation_metrics(
        df, predictors, response
    )
    corr_figs = create_corrheatmapfigs(corr_cat_cat, corr_con_cat, corr_con_con)
    corr_cat_cat, corr_con_cat, corr_con_con = convert_to_desired_format(
        corr_cat_cat, corr_con_cat, corr_con_con
    )

    ########################################################################################################

    ##########################CODE FOR 2 PREDICTOR TABLES(Continuous & Categorical Predictors)########################

    # creating dataframe of the columns with names from the predictors
    df_pred = df[predictors]
    # creating dataframe of the columns with names from the response
    df_res = df[response]

    predictor_plots(df_pred, df_res)
    print("df_pred", df_pred.head())
    print("df_res", df_res.head())

    # print("continuous_pred_df",continuous_pred_df)
    # print("catagorical_pred_df",catagorical_pred_df)

    ########################################################################################################

    (
        continuous_pred_df,
        catagorical_pred_df,
        continuous_pred,
        catagorical_pred,
    ) = create_predictor_dfs(df, response)

    cat_cat_brute_force_df = cat_cat_brute_force(catagorical_pred, df)
    cat_con_brute_force_df = cat_con_brute_force(catagorical_pred, continuous_pred, df)
    con_con_brute_force_df = con_con_brute_force(continuous_pred, df)

    # pval_tscore(df_pred, predictors,response)
    #
    # # ////////////////////////////////////////////////////

    for col1 in continuous_pred:
        for col2 in continuous_pred:
            print(col1, col2)
            cont_cont_brute_matrix, mean_diff_2d = brute_force(col1, col2, response, df)
            print("check 1", mean_diff_2d, pd.DataFrame(cont_cont_brute_matrix))
    for col1 in continuous_pred:
        for col2 in catagorical_pred:
            if col1 == "name" or col2 == "name":
                continue
            print(col1, col2)
            cont_cont_brute_matrix, mean_diff_2d = brute_force(col1, col2, response, df)
            print("check 2", mean_diff_2d, pd.DataFrame(cont_cont_brute_matrix))
    for col1 in catagorical_pred:
        for col2 in catagorical_pred:
            if col1 == "name" or col2 == "name":
                continue
            print(col1, col2)
            cont_cont_brute_matrix, mean_diff_2d = brute_force(col1, col2, response, df)
            print("check 3", mean_diff_2d, pd.DataFrame(cont_cont_brute_matrix))

    # # ///////////////////////////////////////////////////////
    for pred in continuous_pred:
        p_t_values(df, pred, response)

    df_con_pred = df[continuous_pred]

    print(df_con_pred.columns)

    impurity_based_feature_importance(df, df_con_pred, response)

    creating_html(
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
