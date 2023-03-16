import sys
from statistics import statistics

from plotly import express as px

from dataset_loader import TestDatasets


# import pandas as pd
def main():
    # creating object of testdatasets class
    datasets = TestDatasets()
    df, predictors, response = datasets.get_test_data_set("boston")

    # print(df.dtypes)
    df["RAD"] = df["RAD"].astype("float64")
    """
    note: I dont know if am doing is right way but I noticed
    that in reference graphs from slide that column type for
    "Rad" was continuous, also to train regression models
    data type "object" is not accepted hence had to convert it to "float".
    """

    # print(df.dtypes)
    predictors_df = df[predictors]

    def to_check_cat_con_pred(columns):
        dict = {}
        for col in columns:
            print(col)
            if columns[col].dtype == "object":
                dict[col] = "== Categorical"
            elif columns[col].dtype in ["int64", "float64"]:
                dict[col] = "== Continuous"
        return dict

    print(to_check_cat_con_pred(predictors_df))
    # print(response)

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
                file=f"plots_cont_cat/lecture_6_cont_response_cont_predictor_{i}_scatter_plot.html",
                include_plotlyjs="cdn",
            )
        return

    ploting_graphs_cont_cat_target_predictors()


if __name__ == "__main__":
    stats = statistics()  # create an instance of the statistics class
    stats.Linear_reg_p_t_stats()  # call the method on the instance
    stats.logit_p_t_stats()
    sys.exit(main())
