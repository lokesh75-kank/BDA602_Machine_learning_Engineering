# import packages

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

# from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def printheader(title):
    print("*" * 80)
    print(title)
    print("*" * 80)


printheader("Dataframe_Stats_Plots")

columns = ["sepal len", "sepal width", "petal len", "petal width", "class"]
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    header=None,
    names=columns,
)
# naming columns
print(df.head())


def main():
    # print(df.head())
    # print(df["sepal len"].dtypes)

    print("Summary Statistics using numpy")
    for col in df.columns:
        if df.dtypes[col] == "float64":
            print(col)
            mean = np.mean(df[col])
            mini = np.min(df[col])
            std = np.std(df[col])
            print(f"mean:{mean},min:{mini},std:{std}")

    print("Using pandas in-build describe function")
    print("summary stats", df.describe())
    # print(np.mean(df["sepal len"]))
    # print(df.iloc[:,0])

    print("Find the quartiles")
    for col in df.columns:
        if df.dtypes[col] == "float64":
            print(f"column name:{col}")
            first_q = np.percentile(df[col], 25)
            sec_q = np.percentile(df[col], 50)
            third_q = np.percentile(df[col], 75)

            print(
                f"1st Quartile: {first_q} \n"
                f"2nd Quartile: {sec_q} \n"
                f"3rd Quartile: {third_q}"
            )

    print("creating different viz to see difference in class")

    # scatter plot
    # scatter_Pt = px.scatter(df,x = df['sepal len'], y = df["sepal width"])

    scatter_fig1 = px.scatter(
        df,
        x="sepal len",
        y="sepal width",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    scatter_fig1.add_scatter(
        x=df["petal len"],
        y=df["petal width"],
        mode="markers",
        name="Petal Measurements",
        showlegend=True,
    )

    scatter_fig2 = px.scatter(
        df,
        x="petal len",
        y="petal width",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    def store_plots(graphtype, name):
        temp = "outpu_graphs_class_diff/" + name + ".html"
        pio.write_html(fig=graphtype, file=temp, include_plotlyjs="cdn")

    store_plots(scatter_fig1, "scatter_sepal_plot")
    store_plots(scatter_fig2, "scatter_petal_plot")

    # scatter_fig1.show()
    # scatter_fig2.show()

    # bar plot
    bar_fig1 = px.bar(
        df,
        x="sepal len",
        y="sepal width",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    bar_fig2 = px.bar(
        df,
        x="petal len",
        y="petal width",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    store_plots(bar_fig1, "bar_sepal")
    store_plots(bar_fig2, "bar_petal")

    # bar_fig1.show()
    # bar_fig2.show()

    # box plot
    box_fig_sepal_len = px.box(
        df,
        x="class",
        y="sepal len",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    box_fig_sepal_width = px.box(
        df,
        x="class",
        y="sepal width",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    box_fig_petal_len = px.box(
        df,
        x="class",
        y="petal len",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    box_fig_petal_width = px.box(
        df,
        x="class",
        y="petal width",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    store_plots(box_fig_sepal_width, "box_sepal_width")
    store_plots(box_fig_sepal_len, "box_sepal_len")
    store_plots(box_fig_petal_width, "box_petal_width")
    store_plots(box_fig_petal_len, "box_petal_len")

    # box_fig_sepal_len.show()
    # box_fig_sepal_width.show()
    # box_fig_petal_len.show()
    # box_fig_petal_width.show()

    # violin plot

    violin_fig_sepal_len = px.violin(
        df,
        x="class",
        y="sepal len",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    violin_fig_sepal_width = px.violin(
        df,
        x="class",
        y="sepal width",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    violin_fig_petal_len = px.violin(
        df,
        x="class",
        y="petal len",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    violin_fig_petal_width = px.violin(
        df,
        x="class",
        y="petal width",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    store_plots(violin_fig_sepal_width, "violin_sepal_width")
    store_plots(violin_fig_sepal_len, "violin_sepal_len")
    store_plots(violin_fig_petal_width, "violin_petal_width")
    store_plots(violin_fig_petal_len, "violin_petal_len")

    # violin_fig_sepal_len.show()
    # violin_fig_sepal_width.show()
    # violin_fig_petal_len.show()
    # violin_fig_petal_width.show()

    # histogram graphs

    histogram_sepal_len = px.histogram(df, x="sepal len", color="class", nbins=30)
    histogram_sepal_width = px.histogram(df, x="sepal width", color="class", nbins=30)
    histogram_petal_len = px.histogram(df, x="petal len", color="class", nbins=30)
    histogram_petal_width = px.histogram(df, x="petal width", color="class", nbins=30)

    store_plots(histogram_sepal_width, "histogram_sepal_width")
    store_plots(histogram_sepal_len, "histogram_sepal_len")
    store_plots(histogram_petal_width, "histogram_petal_width")
    store_plots(histogram_petal_len, "histogram_petal_len")

    # histogram_fig_sepal_len.show()
    # histogram_sepal_width.show()
    # histogram_petal_len.show()
    # histogram_petal_width.show()


printheader("Building Models")

print("Using Random Forest")

print("trying predictions without standard scaler")

x_train = df[["sepal len", "sepal width", "petal len", "petal width"]]
y_train = df["class"]

# train the random model classifier
classifier = RandomForestClassifier(n_estimators=100)
model1 = classifier.fit(x_train, y_train)


# test the model

test_data = {
    "sepal len": [4, 4.9],
    "sepal width": [3.1, 4.2],
    "petal len": [1, 1.3],
    "petal width": [0.3, 0.5],
}

x_test = pd.DataFrame(test_data)
y_pred = model1.predict(x_test)
print(y_pred)

print("trying predictions with standarscaler")

scaler = StandardScaler()
x_train_Std = scaler.fit_transform(x_train)

# train the model
model2 = classifier.fit(x_train_Std, y_train)
y_pred_std = model2.predict(x_test)
print(y_pred_std)


# train the decision tree classifier

print("Using decision tree classifier")
classifier_DT = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
classifier_DT.fit(x_train_Std, y_train)
print(y_pred_std)


printheader("model via pipeline predictions")
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("RandomForest", RandomForestClassifier(n_estimators=100)),
    ]
)

pipeline.fit(x_train, y_train)
y_pred_pipeline = pipeline.predict(x_test)
print(y_pred_pipeline)
print("Pipeline score", pipeline.score(x_train_Std, y_train))

printheader("Difference with Mean of Response Plot")


def store_response_plots(figure, predictors, class_name):
    temp = "output_response_graphs/" + predictors + class_name + ".html"
    pio.write_html(fig=figure, file=temp, include_plotlyjs="cdn")


"""
Knowledge:
CDN parameterallows you to choose how the Plotly JavaScript library is included
in your web page. Specifically, setting the include_plotlyjs parameter to "cdn" 
tells Plotly to load the Plotly JavaScript library from a CDN (Content Delivery Network)
instead of including the library in your web page directly.
"""


def diff_mean_plots(df, predictors, class_name):
    x = predictors + "_bin_"
    y = "_is_" + (predictors.lower()).replace("-", "_")

    df[y] = (df["class"].str.lower() == class_name.lower()).astype(int)

    # df[x] = (pd.cut(df[predictors], bins=10, right=True)).apply(lambda x: x.mid)
    df[x] = pd.cut(df[predictors], bins=10, right=True)
    mid_points = []
    for interval in df[x]:
        mid_points.append(interval.mid)
    df[x] = mid_points

    df_for_histogram = df[x].value_counts().to_frame().reset_index()
    mean_bins = df[[x, y]].groupby(x).mean().reset_index()
    # bar_figure = go.Figure()
    bar_figure = make_subplots(specs=[[{"secondary_y": True}]])

    bar_figure.add_trace(
        go.Bar(x=df_for_histogram["index"], y=df_for_histogram[x], name=predictors),
        secondary_y=False,
    )
    bar_figure.add_trace(
        go.Scatter(
            x=mean_bins[x],
            y=mean_bins[y],
            name=predictors + "bins mean",
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
            name=predictors + "mean",
            mode="lines",
        ),
        secondary_y=True,
    )

    bar_figure.update_layout(
        title="Mean Response Plot of " + predictors,
        yaxis=dict(title=dict(text="Total Population"), side="left"),
        yaxis2=dict(title=dict(text="Response side"), side="right", overlaying="y"),
        xaxis=dict(title=dict(text=predictors + " Bins")),
    )
    # bar_figure.write_html(file=f"{predictors}_{class_name}.html", include_plotlyjs="cdn")

    store_response_plots(bar_figure, predictors, class_name)

    # fig_bar.show()


# print(df['class'].unique())
# for Iri-setosa class
diff_mean_plots(df, "sepal len", "Iris-setosa")
diff_mean_plots(df, "sepal width", "Iris-setosa")
diff_mean_plots(df, "petal len", "Iris-setosa")
diff_mean_plots(df, "petal width", "Iris-setosa")

diff_mean_plots(df, "sepal len", "Iris-versicolor")
diff_mean_plots(df, "sepal width", "Iris-versicolor")
diff_mean_plots(df, "petal len", "Iris-versicolor")
diff_mean_plots(df, "petal width", "Iris-versicolor")

diff_mean_plots(df, "sepal len", "Iris-virginica")
diff_mean_plots(df, "sepal width", "Iris-virginica")
diff_mean_plots(df, "petal len", "Iris-virginica")
diff_mean_plots(df, "petal width", "Iris-virginica")
#
if __name__ == "__main__":
    main()
