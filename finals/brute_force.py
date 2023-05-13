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
    # if isinstance(catagorical_pred, str):
    #     catagorical_pred = [catagorical_pred]  # Convert single string to a list
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
        # corr, abs_val = calculate_corr_con_cat(cat_values, cont_values)
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
        cat_measures = values[f_cat == i]
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


# ================================================================


# def diff_mean_resp_weighted_ranking_cat_cat(list1, list2, df, response_name):
#     uw_mse_list = []
#     uw_mse_list_ans = []
#     w_mse_list = []
#     w_mse_list_ans = []
#     bin_size = 10  # Set bin size to 10
#
#     for cat1_start in range(math.floor(df[list1].values.min()), math.ceil(df[list1].values.max()), bin_size):
#         cat1_end = cat1_start + bin_size
#         for cat2_start in range(math.floor(df[list2].values.min()), math.ceil(df[list2].values.max()), bin_size):
#             cat2_end = cat2_start + bin_size
#             print("check 1")
#
#             my_list = []
#             cat_mean_list = []
#             for i in range(cat1_start, cat1_end):
#                 temp = []
#                 for j in range(cat2_start, cat2_end):
#                     cat_mean = df[(df[list1] >= i) & (df[list1] < i + bin_size) &
#                                   (df[list2] >= j) & (df[list2] < j + bin_size)][response_name].mean()
#                     cat_count = df[(df[list1] >= i) & (df[list1] < i + bin_size) &
#                                    (df[list2] >= j) & (df[list2] < j + bin_size)][response_name].count()
#
#                     print("check 2")
#                     my_list.append([i, j, cat_count, cat_mean])
#                     temp.append(cat_mean)
#                 cat_mean_list.append(temp)
#
#             # Create directory if it doesn't exist
#             if not os.path.exists("Brute_Cat_Cat_Plots"):
#                 os.makedirs("Brute_Cat_Cat_Plots")
#
#             fig = go.Figure(
#                 data=go.Heatmap(
#                     z=cat_mean_list,
#                     x=np.arange(cat1_start, cat1_end),
#                     y=np.arange(cat2_start, cat2_end),
#                     colorscale="RdBu",
#                     zmin=-3.5,
#                     zmax=3,
#                 )
#             )
#             print("check 3")
#
#             file_name = f"Plot_{cat1_start}_{cat1_end}_{cat2_start}_{cat2_end}.html"
#             file_path = os.path.join("Brute_Cat_Cat_Plots", file_name)
#             pio.write_html(fig, file_path, include_plotlyjs="cdn")
#
#             table = pd.DataFrame(my_list, columns=["cat1", "cat2", "BinCount", "BinMean"])
#             pop_mean = df[response_name].mean()
#             table["PopulationMean"] = pop_mean
#             table["MeanSquareDiff"] = [(i - pop_mean) ** 2 for i in table["BinMean"]]
#             # table["PopulationProportion"] = [i / table["BinCount"].sum() for i in table["BinCount"]]
#             table["PopulationProportion"] = [i / table["BinCount"].sum()
#                                              if table["BinCount"].sum() != 0 else 0 for i in
#                                              table["BinCount"]]
#             print("check 3")
#             table["WeightedMSD"] = table["MeanSquareDiff"] * table["PopulationProportion"]
#
#             uw_mse_list.append(table["MeanSquareDiff"].sum() / (len(cat_mean_list) * len(cat_mean_list[0])))
#             w_mse_list.append(table["WeightedMSD"].sum())
#
#             uw_mse_list_ans.append(uw_mse_list[-1])
#             w_mse_list_ans.append(w_mse_list[-1])
#             print("end line")
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
    no_bins = 10
    for cont1 in list1:
        pred1 = df[cont1]
        min_pred1 = pred1.min()
        max_pred1 = pred1.max()
        bin_size1 = (max_pred1 - min_pred1) / no_bins
        for cont2 in list2:
            my_list = []
            pred2 = df[cont2]
            # Y = df[response_name]
            min_pred2 = pred2.min()
            max_pred2 = pred2.max()
            bin_size2 = (max_pred2 - min_pred2) / no_bins
            x1_l = []
            x2_l = []
            x1_u = []
            x2_u = []
            x1_mid = []
            x2_mid = []
            for k in range(no_bins):
                x1_l.append(min_pred1 + (bin_size1 * k))
                x2_l.append(min_pred2 + (bin_size2 * k))
                x1_u.append(min_pred1 + (bin_size1 * (k + 1)))
                x2_u.append(min_pred2 + (bin_size2 * (k + 1)))
                x1_mid.append(
                    (min_pred1 + (bin_size1 * k))
                    + (min_pred1 + (bin_size1 * (k + 1))) / 2
                )
                x2_mid.append(
                    (min_pred2 + (bin_size2 * k))
                    + (min_pred2 + (bin_size2 * (k + 1))) / 2
                )
            cat_mean_list = []
            for i in range(no_bins):
                temp = []
                for j in range(no_bins):
                    # cat_mean = df[(df[cont1] == i) & (df[cont2] == j)][
                    #     response_name
                    # ].mean()
                    # cat_count = df[(df[cont1] == i) & (df[cont2] == j)][
                    #     response_name
                    # ].count()

                    x1_cond = df[cont1].between(x1_l[i], x1_u[i], inclusive="right")
                    x2_cond = df[cont2].between(x2_l[j], x2_u[j], inclusive="right")

                    bin_mean = df[x1_cond & x2_cond][response_name].mean()
                    bin_count = df[x1_cond & x2_cond][response_name].count()

                    my_list.append([i, j, bin_count, bin_mean])
                    temp.append(bin_mean)
                cat_mean_list.append(temp)

            # Create directory if it doesn't exist

            if not os.path.exists("Brute_Con_Con_Plots"):
                os.makedirs("Brute_Con_Con_Plots")

            fig = go.Figure(
                data=go.Heatmap(
                    z=cat_mean_list,
                    x=np.array(no_bins),
                    y=np.array(no_bins),
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
            uw_mse_list.append(table["MeanSquareDiff"].sum() / (no_bins * no_bins))

            w_mse_list.append(table["WeightedMSD"].sum())

            uw_mse_list_ans.append(uw_mse_list[-1])
            w_mse_list_ans.append(w_mse_list[-1])

            # print("here")
            # print(cat1, uw_mse_list[-1], cat2, w_mse_list[-1])
    # uw_mse = sum(uw_mse_list) / len(uw_mse_list)
    # w_mse = sum(w_mse_list) / len(w_mse_list)
    # print("here",uw_mse, w_mse)

    return uw_mse_list_ans, w_mse_list_ans
