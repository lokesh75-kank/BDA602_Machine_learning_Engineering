import itertools
import os
import warnings

import numpy as numpy
import numpy as np
import pandas as pd
import pandas as pandas
import plotly.graph_objs as go
import plotly.io as pio
from scipy import stats


def form_pairs(list1, list2):
    # create all combinations of pairs
    pairs = list(itertools.product(list1, list2))
    all_pairs = pairs
    # create dataframe
    df = pd.DataFrame(all_pairs, columns=["col1", "col2"])
    return df


def cat_cat_brute_force(catagorical_pred, df, response):
    my_df = form_pairs(catagorical_pred, catagorical_pred)

    pearson = []
    abs_pearson = []

    uwe, we = diff_mean_resp_weighted_ranking_cat_cat(
        catagorical_pred, catagorical_pred, df, response
    )

    for index, row in my_df.iterrows():
        # access the values in column A and B for each row
        # print("rows")
        # print(row['col1'], row['col2'])
        corr, abs_corr = calculate_corr_cat_cat(df[row["col1"]], df[row["col2"]])
        pearson.append(corr)
        abs_pearson.append(abs_corr)

    my_df["diff_mean_resp_ranking"] = uwe
    my_df["diff_mean_resp_weighted_ranking"] = we
    my_df["pearson"] = pearson
    my_df["abs_pearson"] = abs_pearson

    ans_df = my_df[my_df["col1"] != my_df["col2"]]

    my_df = my_df.rename(columns={"col1": "cat_1"})
    my_df = my_df.rename(columns={"col2": "cat_2"})
    ans_df = ans_df.rename(columns={"col1": "cat_1"})
    ans_df = ans_df.rename(columns={"col2": "cat_2"})

    return ans_df, my_df


def cat_con_brute_force(catagorical_pred, continuous_pred, df, response):
    my_df = form_pairs(catagorical_pred, continuous_pred)

    corr_list = []
    abs_list = []

    uwe, we = diff_mean_resp_weighted_ranking_cont_cat(
        continuous_pred, catagorical_pred, df, response
    )

    for index, row in my_df.iterrows():
        # access the values in column A and B for each row
        # print("rows")
        # print(row['col1'], row['col2'])

        corr, abs_val = calculate_corr_con_cat(df[row["col1"]], df[row["col2"]])
        corr_list.append(corr)
        abs_list.append(abs_val)

    my_df["diff_mean_resp_ranking"] = uwe
    my_df["diff_mean_resp_weighted_ranking"] = we
    my_df["corr_ratio"] = corr_list
    my_df["abs_corr_ratio"] = abs_list

    ans_df = my_df[my_df["col1"] != my_df["col2"]]

    my_df = my_df.rename(columns={"col1": "cat"})
    my_df = my_df.rename(columns={"col2": "cont"})
    ans_df = ans_df.rename(columns={"col1": "cat"})
    ans_df = ans_df.rename(columns={"col2": "cont"})
    return ans_df, my_df


def con_con_brute_force(continuous_pred, df, response):
    my_df = form_pairs(continuous_pred, continuous_pred)

    pearson = []
    abs_pearson = []

    uwe, we = diff_mean_resp_weighted_ranking_cont_cont(
        continuous_pred, continuous_pred, df, response
    )

    for index, row in my_df.iterrows():
        # access the values in column A and B for each row
        # print("rows")
        # print(row['col1'], row['col2'])
        corr, abs_corr = calculate_pearson_con_con(df[row["col1"]], df[row["col2"]])
        pearson.append(corr)
        abs_pearson.append(abs_corr)

    my_df["diff_mean_resp_ranking"] = uwe
    my_df["diff_mean_resp_weighted_ranking"] = we
    my_df["pearson"] = pearson
    my_df["abs_pearson"] = abs_pearson

    ans_df = my_df[my_df["col1"] != my_df["col2"]]

    my_df = my_df.rename(columns={"col1": "cont_1"})
    my_df = my_df.rename(columns={"col2": "cont_2"})
    ans_df = ans_df.rename(columns={"col1": "cont_1"})
    ans_df = ans_df.rename(columns={"col2": "cont_2"})

    # for i, colname in enumerate(ans_df['cont_1']):
    #     link = "'Continuous_Mean_of_Response_Plot/Mean_of_Response_Plot_"+colname+".html'"
    #     temp = '<a href=' + link + '>' + colname + '</a>'
    #     ans_df.loc[i, 'cont_1'] = temp

    return ans_df, my_df


def calculate_corr_cat_cat(x, y, bias_correction=True, tschuprow=False):

    corr_coeff = numpy.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pandas.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff, abs(corr_coeff)
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff, abs(corr_coeff)
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff, abs(corr_coeff)
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff, abs(corr_coeff)
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff, abs(corr_coeff)


def calculate_corr_con_cat(categories, values):

    # correlation_coef, p_value = kendalltau(val1,val2)

    f_cat, _ = pandas.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)

    abs_val = abs(eta)
    return eta, abs_val


def calculate_pearson_con_con(val1, val2):
    corr, pval = stats.pearsonr(val1, val2)

    abs_corr = abs(corr)
    return corr, abs_corr


def fill_na(data):
    if isinstance(data, pandas.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


# def diff_mean_resp_weighted_ranking_cat_cat(list1, list2, df, response_name):
#     uw_mse_list = []
#     uw_mse_list_ans = []
#     w_mse_list = []
#     w_mse_list_ans = []
#
#     for i in range(len(list1) - 1):  # Iterate over the buckets in list1
#         bin1_start = list1[i]
#         bin1_end = list1[i + 1]
#
#         for j in range(len(list2) - 1):  # Iterate over the buckets in list2
#             bin2_start = list2[j]
#             bin2_end = list2[j + 1]
#
#             my_list = []
#             cat_mean_list = []
#
#             # Filter the dataframe based on bucket ranges
#             filtered_df = df[(df[cat1] >= bin1_start) & (df[cat1] < bin1_end) &
#                              (df[cat2] >= bin2_start) & (df[cat2] < bin2_end)]
#
#             no_bins1 = filtered_df[cat1].unique()
#             no_bins2 = filtered_df[cat2].unique()
#
#             for k in no_bins1:
#                 temp = []
#
#                 for l in no_bins2:
#                     cat_mean = filtered_df[(filtered_df[cat1] == k) &
#                                            (filtered_df[cat2] == l)][response_name].mean()
#
#                     cat_count = filtered_df[(filtered_df[cat1] == k) &
#                                             (filtered_df[cat2] == l)][response_name].count()
#
#                     my_list.append([k, l, cat_count, cat_mean])
#                     temp.append(cat_mean)
#
#                 cat_mean_list.append(temp)
#
#             # Create directory if it doesn't exist
#             if not os.path.exists("Brute_Cat_Cat_Plots"):
#                 os.makedirs("Brute_Cat_Cat_Plots")
#
#             fig = go.Figure(
#                 data=go.Heatmap(
#                     z=cat_mean_list,
#                     x=np.array(no_bins1),
#                     y=np.array(no_bins2),
#                     colorscale="RdBu",
#                     zmin=-3.5,
#                     zmax=3,
#                 )
#             )
#
#             file_name = f"Plot_{bin1_start}_{bin1_end}_{bin2_start}_{bin2_end}.html"
#             file_path = os.path.join("Brute_Cat_Cat_Plots", file_name)
#             pio.write_html(fig, file_path, include_plotlyjs="cdn")
#
#             table = pd.DataFrame(my_list, columns=["cat1", "cat2", "BinCount", "BinMean"])
#             pop_mean = filtered_df[response_name].mean()
#             table["PopulationMean"] = pop_mean
#             table["MeanSquareDiff"] = [((m - pop_mean) ** 2) for m in table["BinMean"]]
#             table["PopulationProportion"] = [n / table["BinCount"].sum() for n in table["BinCount"]]
#             table["WeightedMSD"] = table["MeanSquareDiff"] * table["PopulationProportion"]
#
#             uw_mse_list.append(table["MeanSquareDiff"].sum() / (len(no_bins1) * len(no_bins2)))
#             w_mse_list.append(table["WeightedMSD"].sum())
#
#             uw_mse_list_ans.append(uw_mse_list[-1])
#             w_mse_list_ans.append(w_mse_list[-1])
#
#     return uw_mse_list_ans, w_mse_list_ans

# ==========================================================================================
def diff_mean_resp_weighted_ranking_cat_cat(list1, list2, df, response_name):
    uw_mse_list = []
    uw_mse_list_ans = []
    w_mse_list = []
    w_mse_list_ans = []
    for cat1 in list1:
        for cat2 in list2:
            my_list = []
            no_bins1 = df[cat1].unique()
            no_bins2 = df[cat2].unique()
            cat_mean_list = []
            for i in no_bins1:
                temp = []
                for j in no_bins2:
                    cat_mean = df[(df[cat1] == i) & (df[cat2] == j)][
                        response_name
                    ].mean()
                    cat_count = df[(df[cat1] == i) & (df[cat2] == j)][
                        response_name
                    ].count()

                    # tried this to define range of bins but got errors
                    # no_bins1 = np.linspace(df[cat1].min(), df[cat1].max(), num=11)
                    # no_bins2 = np.linspace(df[cat2].min(), df[cat2].max(), num=11)
                    # cat_mean_list = []
                    # for i in range(len(no_bins1) - 1):
                    #     temp = []
                    #     for j in range(len(no_bins2) - 1):
                    #         cat_mean = df[(df[cat2] >= no_bins1[i]) &
                    #         (df[cat1] < no_bins1[i + 1]) &
                    #                       (df[cat2] >= no_bins2[j]) &
                    #                       (df[cat2] < no_bins2[j + 1])][response_name].mean()
                    #         cat_count = df[(df[cat1] >= no_bins1[i]) & (df[cat1] < no_bins1[i + 1]) &
                    #                        (df[cat2] >= no_bins2[j]) &
                    #                        (df[cat2] < no_bins2[j + 1])][response_name].count()

                    my_list.append([i, j, cat_count, cat_mean])
                    temp.append(cat_mean)
                cat_mean_list.append(temp)
            # print(no_bins1)
            # print(no_bins2)
            # print(cat_mean_list)

            # Create directory if it doesn't exist
            if not os.path.exists("Brute_Cat_Cat_Plots"):
                os.makedirs("Brute_Cat_Cat_Plots")

            fig = go.Figure(
                data=go.Heatmap(
                    z=cat_mean_list,
                    x=np.array(no_bins1),
                    y=np.array(no_bins2),
                    colorscale="RdBu",
                    zmin=-3.5,
                    zmax=3,
                )
            )

            file_name = f"Plot_{cat1}_{cat2}.html"
            file_path = os.path.join("Brute_Cat_Cat_Plots", file_name)
            pio.write_html(fig, file_path, include_plotlyjs="cdn")

            # fig.show()

            table = pd.DataFrame(
                my_list, columns=["cat1", "cat2", "BinCount", "BinMean"]
            )
            pop_mean = df[response_name].mean()
            table["PopulationMean"] = pop_mean
            table["MeanSquareDiff"] = [((i - pop_mean) ** 2) for i in table["BinMean"]]
            table["PopulationProportion"] = [
                i / table["BinCount"].sum() for i in table["BinCount"]
            ]
            table["WeightedMSD"] = (
                table["MeanSquareDiff"] * table["PopulationProportion"]
            )

            uw_mse_list.append(
                table["MeanSquareDiff"].sum() / (len(no_bins1) * len(no_bins2))
            )
            w_mse_list.append(table["WeightedMSD"].sum())

            uw_mse_list_ans.append(uw_mse_list[-1])
            w_mse_list_ans.append(w_mse_list[-1])

            # print("here")
            # print(cat1, uw_mse_list[-1], cat2, w_mse_list[-1])
    # uw_mse = sum(uw_mse_list) / len(uw_mse_list)
    # w_mse = sum(w_mse_list) / len(w_mse_list)
    # print("here",uw_mse, w_mse)

    return uw_mse_list_ans, w_mse_list_ans


def diff_mean_resp_weighted_ranking_cont_cat(list1, list2, df, response_name):
    uw_mse_list = []
    uw_mse_list_ans = []
    w_mse_list = []
    w_mse_list_ans = []
    for cont in list1:
        for cat in list2:
            my_list = []
            no_bins1 = df[cont].unique()
            no_bins2 = df[cat].unique()
            cat_mean_list = []
            for i in no_bins1:
                temp = []
                for j in no_bins2:
                    cat_mean = df[(df[cont] == i) & (df[cat] == j)][
                        response_name
                    ].mean()
                    cat_count = df[(df[cont] == i) & (df[cat] == j)][
                        response_name
                    ].count()

                    # no_bins1 = np.linspace(df[cont].min(), df[cont].max(), num=11)
                    # no_bins2 = np.linspace(df[cat].min(), df[cat].max(), num=11)
                    # cat_mean_list = []
                    # for i in range(len(no_bins1) - 1):
                    #     temp = []
                    #     for j in range(len(no_bins2) - 1):
                    #         cat_mean = df[(df[cont] >= no_bins1[i]) &
                    #         (df[cont] < no_bins1[i + 1]) &
                    #                       (df[cat] >= no_bins2[j]) &
                    #                       (df[cat] < no_bins2[j + 1])][response_name].mean()
                    #         cat_count = df[(df[cont] >= no_bins1[i]) &
                    #         (df[cont] < no_bins1[i + 1]) &
                    #                        (df[cat] >= no_bins2[j]) &
                    #                        (df[cat] < no_bins2[j + 1])][response_name].count()

                    my_list.append([i, j, cat_count, cat_mean])
                    temp.append(cat_mean)
                cat_mean_list.append(temp)
            # print(no_bins1)
            # print(no_bins2)
            # print(cat_mean_list)

            # Create directory if it doesn't exist

            if not os.path.exists("Brute_Con_Cat_Plots"):
                os.makedirs("Brute_Con_Cat_Plots")

            fig = go.Figure(
                data=go.Heatmap(
                    z=cat_mean_list,
                    x=np.array(no_bins1),
                    y=np.array(no_bins2),
                    colorscale="RdBu",
                    zmin=-3.5,
                    zmax=3,
                )
            )

            file_name = f"Plot_{cont}_{cat}.html"
            file_path = os.path.join("Brute_Con_Cat_Plots", file_name)
            pio.write_html(fig, file_path, include_plotlyjs="cdn")

            # fig.show()

            table = pd.DataFrame(
                my_list, columns=["cat1", "cat2", "BinCount", "BinMean"]
            )
            pop_mean = df[response_name].mean()
            table["PopulationMean"] = pop_mean
            table["MeanSquareDiff"] = [((i - pop_mean) ** 2) for i in table["BinMean"]]
            table["PopulationProportion"] = [
                i / table["BinCount"].sum() for i in table["BinCount"]
            ]
            table["WeightedMSD"] = (
                table["MeanSquareDiff"] * table["PopulationProportion"]
            )

            uw_mse_list.append(
                table["MeanSquareDiff"].sum() / (len(no_bins1) * len(no_bins2))
            )
            w_mse_list.append(table["WeightedMSD"].sum())

            uw_mse_list_ans.append(uw_mse_list[-1])
            w_mse_list_ans.append(w_mse_list[-1])

            # print(cat1, uw_mse_list[-1], cat2, w_mse_list[-1])
    # uw_mse = sum(uw_mse_list) / len(uw_mse_list)
    # w_mse = sum(w_mse_list) / len(w_mse_list)

    return uw_mse_list_ans, w_mse_list_ans


def diff_mean_resp_weighted_ranking_cont_cont(list1, list2, df, response_name):
    uw_mse_list = []
    uw_mse_list_ans = []
    w_mse_list = []
    w_mse_list_ans = []
    for cont1 in list1:
        for cont2 in list2:
            my_list = []
            no_bins1 = df[cont1].unique()
            # print("no_bins01\n",no_bins1)
            no_bins2 = df[cont2].unique()
            cat_mean_list = []
            for i in no_bins1:
                temp = []
                for j in no_bins2:
                    cat_mean = df[(df[cont1] == i) & (df[cont2] == j)][
                        response_name
                    ].mean()
                    cat_count = df[(df[cont1] == i) & (df[cont2] == j)][
                        response_name
                    ].count()

                    # tried this to define range of bins but got errors
                    # no_bins1 = np.linspace(df[cont1].min(), df[cont1].max(), num=11)
                    # no_bins2 = np.linspace(df[cont2].min(), df[cont2].max(), num=11)
                    # cat_mean_list = []
                    # for i in range(len(no_bins1) - 1):
                    #     temp = []
                    #     for j in range(len(no_bins2) - 1):
                    #         cat_mean = df[(df[cont1] >= no_bins1[i]) &
                    #         (df[cont1] < no_bins1[i + 1]) &
                    #                       (df[cont2] >= no_bins2[j]) &
                    #                       (df[cont2] < no_bins2[j + 1])][response_name].mean()
                    #         cat_count = df[(df[cont1] >= no_bins1[i]) &
                    #         (df[cont1] < no_bins1[i + 1]) &
                    #                        (df[cont2] >= no_bins2[j]) &
                    #                        (df[cont2] < no_bins2[j + 1])][response_name].count()

                    my_list.append([i, j, cat_count, cat_mean])
                    temp.append(cat_mean)
                cat_mean_list.append(temp)
            # print(no_bins1)
            # print(no_bins2)
            # print(cat_mean_list)

            # Create directory if it doesn't exist

            if not os.path.exists("Brute_Con_Con_Plots"):
                os.makedirs("Brute_Con_Con_Plots")

            fig = go.Figure(
                data=go.Heatmap(
                    z=cat_mean_list,
                    x=np.array(no_bins1),
                    y=np.array(no_bins2),
                    colorscale="RdBu",
                    zmin=-3.5,
                    zmax=3,
                )
            )

            file_name = f"Plot_{cont1}_{cont2}.html"
            file_path = os.path.join("Brute_Con_Con_Plots", file_name)
            pio.write_html(fig, file_path, include_plotlyjs="cdn")

            # fig.show()

            table = pd.DataFrame(
                my_list, columns=["cat1", "cat2", "BinCount", "BinMean"]
            )
            pop_mean = df[response_name].mean()
            table["PopulationMean"] = pop_mean
            table["MeanSquareDiff"] = [((i - pop_mean) ** 2) for i in table["BinMean"]]
            table["PopulationProportion"] = [
                i / table["BinCount"].sum() for i in table["BinCount"]
            ]
            table["WeightedMSD"] = (
                table["MeanSquareDiff"] * table["PopulationProportion"]
            )

            uw_mse_list.append(
                table["MeanSquareDiff"].sum() / (len(no_bins1) * len(no_bins2))
            )
            w_mse_list.append(table["WeightedMSD"].sum())

            uw_mse_list_ans.append(uw_mse_list[-1])
            w_mse_list_ans.append(w_mse_list[-1])

            # print("here")
            # print(cat1, uw_mse_list[-1], cat2, w_mse_list[-1])
    # uw_mse = sum(uw_mse_list) / len(uw_mse_list)
    # w_mse = sum(w_mse_list) / len(w_mse_list)
    # print("here",uw_mse, w_mse)

    return uw_mse_list_ans, w_mse_list_ans
