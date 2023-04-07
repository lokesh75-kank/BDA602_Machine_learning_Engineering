import itertools
import warnings

import numpy as numpy
import numpy as np
import pandas as pd
import pandas as pandas
from scipy import stats


def form_pairs(list1, list2):
    # create all combinations of pairs
    pairs = list(itertools.product(list1, list2))
    # create reversed pairs as well
    # rev_pairs = [(y, x) for x, y in pairs]
    # combine both pairs and reverse pairs
    all_pairs = pairs
    # create dataframe
    df = pd.DataFrame(all_pairs, columns=["col1", "col2"])
    return df


def cat_cat_brute_force(catagorical_pred, df):
    my_df = form_pairs(catagorical_pred, catagorical_pred)

    pearson = []
    # abs_pearson = []

    for index, row in my_df.iterrows():
        # access the values in column A and B for each row
        # print("rows")
        # print(row['col1'], row['col2'])
        corr = calculate_corr_cat_cat(df[row["col1"]], df[row["col2"]])
        pearson.append(corr)
        # abs_pearson.append(abs_corr)

    my_df["pearson"] = pearson
    # my_df['abs_pearson'] = abs_pearson

    my_df = my_df[my_df["col1"] != my_df["col2"]]

    return my_df


def cat_con_brute_force(catagorical_pred, continuous_pred, df):
    my_df = form_pairs(catagorical_pred, continuous_pred)

    print("check my_df", my_df)

    corr_list = []

    for index, row in my_df.iterrows():
        # access the values in column A and B for each row
        # print("rows")
        # print(row['col1'], row['col2'])

        corr = calculate_corr_con_cat(df[row["col1"]], df[row["col2"]])
        corr_list.append(corr)

    my_df["corr_list"] = corr_list
    # my_df['abs_pearson'] = abs_pearson

    my_df = my_df[my_df["col1"] != my_df["col2"]]
    return my_df


def con_con_brute_force(continuous_pred, df):
    my_df = form_pairs(continuous_pred, continuous_pred)

    pearson = []
    abs_pearson = []

    for index, row in my_df.iterrows():
        # access the values in column A and B for each row
        # print("rows")
        # print(row['col1'], row['col2'])
        corr, abs_corr = calculate_pearson_con_con(df[row["col1"]], df[row["col2"]])
        pearson.append(corr)
        abs_pearson.append(abs_corr)

    my_df["pearson"] = pearson
    my_df["abs_pearson"] = abs_pearson

    my_df = my_df[my_df["col1"] != my_df["col2"]]

    return my_df


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
    return eta


def calculate_pearson_con_con(val1, val2):
    corr, pval = stats.pearsonr(val1, val2)

    abs_corr = abs(corr)
    return corr, abs_corr


def fill_na(data):
    if isinstance(data, pandas.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


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
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def is_continuous(data, col):
    # This Function takes in a column of a pandas data frame and returns a boolean depending on if the column variables
    # are continuous or not.

    if (
        len(data[col].unique()) <= 0.1 * len(data[col])
        or data[col].dtype == "O"
        or data[col].dtype == str
    ):
        return False
    return True


def brute_force(pred1, pred2, resp, data):

    copy_data = data.copy()
    resp_mean = data[resp].mean()

    if is_continuous(data, pred1):
        copy_data["bins1"] = pd.cut(copy_data[pred1], 10, labels=False)
        if is_continuous(data, pred2):
            copy_data["bins2"] = pd.cut(copy_data[pred2], 10, labels=False)
            calc_matrix = np.zeros((10, 10))
            pop_ratio_matrix = np.zeros_like((calc_matrix))
            for x_1, val1 in enumerate(copy_data["bins1"].unique()):
                for x_2, val2 in enumerate(copy_data["bins2"].unique()):
                    bin_mean = copy_data[resp][
                        (copy_data["bins1"] == x_1) & (copy_data["bins2"] == x_2)
                    ].mean()
                    pop_ratio = (
                        len(copy_data[resp][copy_data["bins1"] == x_1])
                        / len(copy_data[resp])
                        * len(copy_data[resp][copy_data["bins2"] == x_2])
                        / len(copy_data[resp])
                    )
                    pop_ratio_matrix[x_1, x_2] = pop_ratio
                    calc_matrix[x_1, x_2] = bin_mean
        else:
            calc_matrix = np.zeros((10, len(copy_data[pred2].unique())))
            pop_ratio_matrix = np.zeros_like((calc_matrix))
            for x_1, val1 in enumerate(copy_data["bins1"].unique()):
                for x_2, val2 in enumerate(copy_data[pred2].unique()):
                    bin_mean = copy_data[resp][
                        (copy_data["bins1"] == x_1) & (copy_data[pred2] == val2)
                    ].mean()
                    pop_ratio = (
                        len(copy_data[resp][copy_data["bins1"] == x_1])
                        / len(copy_data[resp])
                        * len(copy_data[resp][copy_data[pred2] == val2])
                        / len(copy_data[resp])
                    )
                    pop_ratio_matrix[x_1, x_2] = pop_ratio
                    calc_matrix[x_1, x_2] = bin_mean
    else:
        if is_continuous(data, pred2):
            calc_matrix = np.zeros((len(copy_data[pred1].unique()), 10))
            pop_ratio_matrix = np.zeros_like((calc_matrix))
            copy_data["bins2"] = pd.cut(copy_data[pred2], 10, labels=False)
            for x_1, val1 in enumerate(copy_data[pred1].unique()):
                for x_2, val2 in enumerate(copy_data["bins2"].unique()):
                    bin_mean = copy_data[resp][
                        (copy_data[pred1] == val1) & (copy_data["bins2"] == x_2)
                    ].mean()
                    pop_ratio = (
                        len(copy_data[resp][copy_data[pred1] == val1])
                        / len(copy_data[resp])
                        * len(copy_data[resp][copy_data["bins2"] == x_2])
                        / len(copy_data[resp])
                    )
                    pop_ratio_matrix[x_1, x_2] = pop_ratio
                    calc_matrix[x_1, x_2] = bin_mean
        else:
            calc_matrix = np.zeros(
                (len(copy_data[pred1].unique()), len(copy_data[pred2].unique()))
            )
            pop_ratio_matrix = np.zeros_like((calc_matrix))
            for x_1, val1 in enumerate(copy_data[pred1].unique()):
                for x_2, val2 in enumerate(copy_data[pred2].unique()):
                    bin_mean = copy_data[resp][
                        (copy_data[pred1] == val1) & (copy_data[pred2] == val2)
                    ].mean()
                    pop_ratio = (
                        len(copy_data[resp][copy_data[pred1] == val1])
                        / len(copy_data[resp])
                        * len(copy_data[resp][copy_data[pred2] == val2])
                        / len(copy_data[resp])
                    )
                    pop_ratio_matrix[x_1, x_2] = pop_ratio
                    calc_matrix[x_1, x_2] = bin_mean

    calc_matrix = np.nan_to_num(calc_matrix)

    avg_mean_diff = (((calc_matrix - resp_mean) ** 2) * pop_ratio_matrix).sum()

    return calc_matrix, avg_mean_diff

    # make a new table where each box is ((value-popmean)^2) * bin_pop/tot_pop
