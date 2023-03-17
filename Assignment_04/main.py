import sys
from statistics import statistics
import plotly.graph_objs as go
from plotly import express as px
from plotly.subplots import make_subplots
from dataset_loader import TestDatasets
import pandas as pd


def main():
    # creating object of testdatasets class
    datasets = TestDatasets()
    df, predictors, response = datasets.get_test_data_set("boston")
    # print(df.describe())
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
    # print(predictors_df)

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
    class_name = "target"
    print(type(class_name))
    print(class_name)

    # def diff_mean_plots(df, predictors_df,class_name):
    #     for i in predictors_df:
    #         x = i + "_bin_"
    #         y = "_is_" + (i.lower())
    #         # df[y] = (df[class_name].astype(str).str.lower() == df[class_name].astype(str).lower()).astype(int)
    #         df[y] = (df[class_name].astype(str).str.lower() == df[class_name].astype(str).str.lower()).astype(int)
    #
    #         # df[x] = (pd.cut(df[predictors], bins=10, right=True)).apply(lambda x: x.mid)
    #         df[x] = pd.cut(df[i], bins=10, right=True)
    #         mid_points = []
    #         for interval in df[x]:
    #             mid_points.append(interval.mid)
    #         df[x] = mid_points
    #
    #         df_for_histogram = df[x].value_counts().to_frame().reset_index()
    #         df[x + '_midpoint'] = df[x].apply(lambda x: x.mid)
    #
    #         # mean_bins = df[[x, y]].groupby(x).mean().reset_index()
    #         # bar_figure = go.Figure()
    #         mean_bins = df[[x + '_midpoint', y]].groupby(x + '_midpoint').mean().reset_index()
    #
    #         bar_figure = make_subplots(specs=[[{"secondary_y": True}]])
    #
    #         bar_figure.add_trace(
    #             go.Bar(x=df_for_histogram["index"], y=df_for_histogram[x], name=i),
    #             secondary_y=False,
    #         )
    #         bar_figure.add_trace(
    #             go.Scatter(
    #                 x=mean_bins[x + '_midpoint'],
    #                 y=mean_bins[y],
    #                 name=i + " bins mean",
    #                 mode="lines + markers",
    #                 marker=dict(color="crimson"),
    #             ),
    #             secondary_y=True,
    #         )
    #         # overall avg of graph
    #         bar_figure.add_trace(
    #             go.Scatter(
    #                 x=mean_bins[x],
    #                 y=[df[y].mean()] * len(mean_bins[x]),
    #                 name=i + "mean",
    #                 mode="lines",
    #             ),
    #             secondary_y=True,
    #         )
    #
    #         bar_figure.update_layout(
    #             title=f"Binned difference with mean of response vs{i} bin",
    #             yaxis=dict(title=dict(text="Response"), side="left"),
    #             yaxis2=dict(title=dict(text="Response side"), side="right", overlaying="y"),
    #             xaxis=dict(title=dict(text= i + " Bins")),
    #         )
    #         bar_figure.write_html(file=f"Diff_mean_response/diff_mean_response{i}_{class_name}.html", include_plotlyjs="cdn")
    #         bar_figure.show()
    # diff_mean_plots(df, predictors_df,"target")

    def diff_mean_plots(df, predictors, class_name):
        for predictor in predictors:
            x = predictor + "_bin_"
            y = "_is_" + predictor.lower()
            df[y] = (df[class_name].astype(str).str.lower() == predictor.lower()).astype(int)
            if df[predictor].dtype == 'object':
                df[x] = df[predictor]
            else:
                df[x] = pd.cut(df[predictor], bins=10, right=True)

                # print(df[x].dtype())
        #         mid_points = []
        #         for interval in df[x]:
        #             mid_points.append(interval.mid)
        #         df[x] = mid_points
        #     print(df.columns)
        #     ## Calculate unweighted means
        # df[x + '_midpoint'] = df[x].mean()
        # mean_bins = df[[x+ '_midpoint', y]].groupby(x+ '_midpoint').mean().reset_index()
        # # Calculate weighted means
        # counts = df[x].value_counts().to_frame().reset_index()
        # counts.columns = [x + '_midpoint', 'count']
        # counts = counts.merge(df[[x + '_midpoint',y]], on=x+ '_midpoint' )
        #
        # weighted_means = counts.groupby(x + '_midpoint')[y].apply(lambda x: (x * counts['count']).sum() / counts['count'].sum()).reset_index()
        # weighted_means.columns = [x + '_midpoint', y]
        # #
        # bar_figure = make_subplots(specs=[[{"secondary_y": True}]])
        # # #
        # # Add unweighted mean line
        # bar_figure.add_trace(
        #     go.Scatter(
        #         x=mean_bins[x + '_midpoint'],
        #         y=mean_bins[y],
        #         name="Unweighted",
        #         mode="lines + markers",
        #         marker=dict(color="crimson"),
        #     ),
        #     secondary_y=True,
        # )
        # # #
        # # Add weighted mean line
        # bar_figure.add_trace(
        #     go.Scatter(
        #         x=weighted_means[x + '_midpoint'],
        #         y=weighted_means[y],
        #         name="Weighted",
        #         mode="lines + markers",
        #         marker=dict(color="blue"),
        #     ),
        #     secondary_y=True,
        # )
        # # #
        # df_for_histogram = df[x].value_counts().to_frame().reset_index()
        # # print(df.columns)
        # bar_figure.add_trace(
        #     go.Bar(x=df_for_histogram["index"], y=df_for_histogram[x], name=predictor),
        #     secondary_y=False,
        # )
        # # #
        # bar_figure.add_trace(
        #     go.Scatter(
        #         x=mean_bins[x + '_midpoint'],
        #         y=[df[y].mean()] * len(mean_bins[x + '_midpoint']),
        #         name="Overall mean",
        #         mode="lines",
        #     ),
        #     secondary_y=True,
        # )
        # # #
        # bar_figure.update_layout(
        #     title=f"Binned difference with mean of response vs {predictor} bin",
        #     yaxis=dict(title=dict(text="Response"), side="left"),
        #     yaxis2=dict(title=dict(text="Response side"), side="right", overlaying="y"),
        #     xaxis=dict(title=dict(text=predictor + " Bins")),
        # )
        # bar_figure.write_html(
        #     file=f"Diff_mean_response/diff_mean_response_{predictor}_{class_name}.html",
        #     include_plotlyjs="cdn"
        # )
                mid_points = []
                for interval in df[x]:
                    mid_points.append(interval.mid)
                df[x] = mid_points

            df[x + '_midpoint'] = df[x].mean()
            df_for_histogram = df[x].value_counts().to_frame().reset_index()
            mean_bins = df[[x, y]].groupby(x).mean().reset_index()

            #Calculate weighted means
            counts = df[x].value_counts().to_frame().reset_index()
            counts.columns = [x + '_midpoint', 'count']
            counts = counts.merge(df[[x + '_midpoint',y]], on=x+ '_midpoint' )

            weighted_means = counts.groupby(x + '_midpoint')[y].apply(lambda x: (x * counts['count']).sum() / counts['count'].sum()).reset_index()
            weighted_means.columns = [x + '_midpoint', y]
            #
            bar_figure = make_subplots(specs=[[{"secondary_y": True}]])

            bar_figure.add_trace(
                go.Bar(x=df_for_histogram["index"], y=df_for_histogram[x], name=predictor),
                secondary_y=False,
            )

            # Add unweighted mean line
            bar_figure.add_trace(
                go.Scatter(
                    x=mean_bins[x],
                    y=mean_bins[y],
                    name="Unweighted",
                    mode="lines + markers",
                    marker=dict(color="crimson"),
                ),
                secondary_y=True,
            )

            # Add weighted mean line
            bar_figure.add_trace(
                go.Scatter(
                    x=weighted_means[x + '_midpoint'],
                    y=weighted_means[y],
                    name="Weighted",
                    mode="lines + markers",
                    marker=dict(color="blue"),
                ),
                secondary_y=True,
            )

            bar_figure.add_trace(
                go.Scatter(
                    x=mean_bins[x],
                    y=mean_bins[y],
                    name=predictor + "bins mean",
                    mode="lines + markers",
                    marker=dict(color="crimson"),
                ),
                secondary_y=True,
            )

            # overall avg of graph
            bar_figure.add_trace(
                go.Scatter(
                    x=mean_bins[x],
                    y=[df[y].mean()] * len(mean_bins[x]),
                    name=predictor + "mean",
                    mode="lines",
                ),
                secondary_y=True,
            )

            bar_figure.update_layout(
                title="Mean Response Plot of " + predictor,
                yaxis=dict(title=dict(text="Response(Target)"), side="left"),
                yaxis2=dict(title=dict(text="Response side"), side="right", overlaying="y"),
                xaxis=dict(title=dict(text=predictor + " Bins")),
            )

            bar_figure.show()
            bar_figure.write_html(file=f"Diff_mean_response/diff_mean_response{predictor}_{class_name}.html",
                                  include_plotlyjs="cdn")

    diff_mean_plots(df, predictors_df,"target")







if __name__ == "__main__":
    stats = statistics()  # create an instance of the statistics class
    stats.Linear_reg_p_t_stats()  # call the method on the instance
    stats.logit_p_t_stats()
    sys.exit(main())
