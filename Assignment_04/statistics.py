import statsmodels.api
from dataset_loader import TestDatasets
from plotly import express as px

# from sklearn import datasets


class statistics:
    def Linear_reg_p_t_stats(self):
        datasets = TestDatasets()
        df, predictors, response = datasets.get_test_data_set("boston")
        df["RAD"] = df["RAD"].astype("float64")
        print(df.dtypes)
        predictors_df = df[predictors]
        predictors_df = predictors_df.drop("CHAS", axis=1)
        # as CHAS is categorical predictor, hence excluding

        x = predictors_df
        y = df["target"]

        for idx, col in enumerate(x.columns):
            feature_name = col
            pred = statsmodels.api.add_constant(x[col])
            linear_regression_model = statsmodels.api.OLS(y, pred)
            linear_regression_model_fitted = linear_regression_model.fit()
            # print(f"Variable: {feature_name}")
            # print(linear_regression_model_fitted.summary())
            """
            In this code, statsmodels.api.add_constant is used to add a
            constant (i.e., an intercept term) to the column data.
            Adding a constant to the predictor is important
            because it ensures that the regression line intercepts the y-axis at the correct point,
             even if the data has a mean value of zero.
            """
            # Get the stats
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
            # Plot the figure
            fig = px.scatter(data_frame=df, x=col, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title="target",
            )
            # fig.show()
            fig.write_html(
                file=f"plots_stats_p_t_linear/{col}.html", include_plotlyjs="cdn"
            )
        return

    def logit_p_t_stats(self):
        datasets = TestDatasets()
        df, predictors, response = datasets.get_test_data_set("boston")
        df["RAD"] = df["RAD"].astype("float64")
        df["target"] = df["target"].apply(lambda x: 1 if x > df["target"].mean() else 0)
        # print(df["target"])
        predictors_df = df[predictors]
        predictors_df = predictors_df.drop("CHAS", axis=1)
        # as CHAS is categorical predictor, hence excluding

        x = predictors_df
        y = df["target"]
        # print("y:",y)

        for idx, col in enumerate(x.columns):
            feature_name = col
            logit = statsmodels.api.Logit(y, statsmodels.api.add_constant(x[col]))
            logit_fitted = logit.fit()

            # Get the stats
            t_value = round(logit_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logit_fitted.pvalues[1])

            # Plot the figure
            fig = px.scatter(data_frame=df, x=col, y="target", trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title="target",
            )
            # fig.show()
            fig.write_html(
                file=f"plots_stats_p_t_logit/{col}.html", include_plotlyjs="cdn"
            )
        return
