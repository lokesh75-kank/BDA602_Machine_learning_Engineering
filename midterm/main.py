import sys

from dataset_loader import TestDatasets

from midterm.brute_force import (
    cat_cat_brute_force,
    cat_con_brute_force,
    con_con_brute_force,
)
from midterm.calculate_correlation_metrics import create_corrheatmapfigs
from midterm.creating_html import creating_html
from midterm.predictors_tables import create_predictor_dfs, predictor_plots


def main():

    # taking sample dataframe for testing purpose
    test_datasets = TestDatasets()
    dataset_name = input("TYPE NAME OF A DATASET: ")
    df, predictors, response = test_datasets.get_test_data_set(dataset_name.lower())
    # df, predictors, response = test_datasets.get_test_data_set("tips")

    # CODE FOR 2 PREDICTOR TABLES(Continuous & Categorical Predictors)

    # creating dataframe of the columns with names from the predictors
    df_pred = df[predictors]
    # creating dataframe of the columns with names from the response
    df_res = df[response]

    # this function saves all the plts in the directories dynamically, also returns
    # list of continuous predictors and catagorical predictors
    continuous_pred, catagorical_pred = predictor_plots(df_pred, df_res)

    # function gives complete 2 dfs to convert to html
    (
        continuous_pred_df,
        catagorical_pred_df,
    ) = create_predictor_dfs(df, response, continuous_pred, catagorical_pred)

    # BRUTE FORCE TABLES
    # gives complete brute force dfs(with and without dropping the same column names)

    """I tried calculations for the diff_mean_resp_ranking, diff_mean_resp_weighted_ranking
    in the brute force table.The calculations for categorical/categorical are accurate,
    but somehow the values for other two tables are not accurate.
    I have also tried plotting the heatmaps.They are accurate but,I couldn't linked them to the dataframe
    because i got few errors.Hence, I have stored them in a folder. """
    cat_cat_brute_force_df, cat_cat_complete = cat_cat_brute_force(
        catagorical_pred, df, response
    )
    cat_con_brute_force_df, cat_con_complete = cat_con_brute_force(
        catagorical_pred, continuous_pred, df, response
    )
    con_con_brute_force_df, con_con_complete = con_con_brute_force(
        continuous_pred, df, response
    )

    # CODE FOR CORRELATION 3 CORRELATION MATRICES AND 3 PLOTS
    # extract desired columns
    corr_cat_cat = cat_cat_complete[["cat_1", "cat_2", "pearson"]]
    corr_con_cat = cat_con_complete[["cat", "cont", "corr_ratio"]]
    corr_con_con = con_con_complete[["cont_1", "cont_2", "pearson"]]

    # create their heatmaps
    corr_figs = create_corrheatmapfigs(corr_cat_cat, corr_con_cat, corr_con_con)

    # df_con_pred = df[continuous_pred]

    # impurity_based_feature_importance(df, df_con_pred, response)

    # converts all the dataframes to html
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
