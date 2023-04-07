import os

import plotly.io as pio
import plotly.offline as pyo


def creating_html(
    corr_cat_cat,
    corr_con_cat,
    corr_con_con,
    figs,
    continuous_pred_df,
    catagorical_pred_df,
    cat_cat_brute_force_df,
    cat_con_brute_force_df,
    con_con_brute_force_df,
):
    with open("midterm.html", "w") as f:
        f.write("<html>\n")
        f.write("<head>\n")
        f.write("<title>Midterm</title>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write("<h1>Welcome to BDA 602 ML Midterm solution by Lokesh Kank</h1>\n")
        f.write("<h2>MPG</h2>\n")
        f.write("<h3>Continuous Predictors</h3>\n")
        # table
        f.write(continuous_pred_df.to_html())

        f.write("<h3>Categorical Predictors</h3>\n")
        # table
        f.write(catagorical_pred_df.to_html())
        ############################################################################################################33
        f.write("<h3>Categorical/Categorical Correlations</h3>\n")
        # plot heatmap
        plot_div = pio.to_html(
            figs[0], full_html=False, auto_play=False, include_plotlyjs="cdn"
        )
        f.write(plot_div)
        f.write(corr_cat_cat.head().to_html() + "\n\n")

        ############################################################################################################33

        f.write("<h3>Categorical/Continuous Correlations</h3>\n")
        # plot heatmap
        plot_div = pio.to_html(
            figs[1], full_html=False, auto_play=False, include_plotlyjs="cdn"
        )
        f.write(plot_div)
        f.write(corr_con_cat.head().to_html() + "\n\n")

        ############################################################################################################33

        f.write("<h3>Continuous/Continuous Correlations</h3>\n")
        # plot heatmap
        plot_div = pio.to_html(
            figs[2], full_html=False, auto_play=False, include_plotlyjs="cdn"
        )
        f.write(plot_div)
        f.write(corr_con_con.to_html() + "\n\n")

        ############################################################################################################33

        # brute force
        f.write('<h3>"Categorical/Categorical - Brute Force" Table</h3>\n')
        f.write(cat_cat_brute_force_df.to_html())
        # table

        f.write('<h3>"Categorical/Continuous - Brute Force</h3>\n')
        # table
        f.write(cat_con_brute_force_df.to_html())

        f.write('<h3>"Continuous/Continuous - Brute Force</h3>\n')
        # table
        f.write(con_con_brute_force_df.to_html())

        f.write("</body>\n")
        f.write("</html>\n")
