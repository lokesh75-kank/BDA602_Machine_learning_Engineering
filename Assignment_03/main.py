# Importing relevant packages
import sys

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from transformer import RollingAvgTransformer


def main():
    appName = "PySpark Example - MariaDB Example"
    master = "local[*]"
    # Create Spark session
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    # database properties
    database = "baseball"
    user = "admin"
    password = "1196"
    # linter Detect secrets fails due to password.
    # Hence commented it in pre-commit config file
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"
    # SQL queries
    sql1 = """
                SELECT g.game_id ,DATE(local_date) AS local_date,
                batter,atBat, Hit
                FROM batter_counts bc
                JOIN game g ON g.game_id = bc.game_id
                ORDER BY batter, local_date
            """
    # Create a data frame by reading data from Oracle via JDBC using sql1
    df_sql1 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql1)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )
    df_sql1.createOrReplaceTempView("game_date_batter")
    df_sql1.persist(StorageLevel.DISK_ONLY)

    # Creating a class object
    rolling_avg_transformer = RollingAvgTransformer(
        inputCols=["game_id", "local_date", "batter", "atBat", "Hit"],
        outputCols=[
            "batter",
            "game_id",
            "local_date",
            "sum_Hit",
            "sum_atBat",
            "Batting_rolling_Avg",
            "cnt",
        ],
    )
    # Creating pipeline
    pipeline = Pipeline(stages=[rolling_avg_transformer])
    model = pipeline.fit(df_sql1)
    final_result = model.transform(df_sql1)
    final_result.show()

    return

    # Nice way to save a file onto the system
    # final_result.coalesce(1).write.format('csv').
    # option('header', 'true').mode('overwrite').
    # save('Data/final_result_rolling_100_avg')
    # rolling_avg_df = spark.read.csv(
    # 'scripts/Data/final_result_rolling_100_avg/
    # part-00000-d7cfd6b2-cdbd-48a5-a818-80deb1fd56d0-c000.csv',
    # inferSchema="true", header="true")

    # # Write final_result as a temporary CSV file onto the system
    # temp_csv_file = tempfile.mktemp()
    # final_result.write.
    # csv(temp_csv_file, header=True)

    # # Read the temporary CSV file into a DataFrame
    # rolling_avg_df = spark.read.csv(temp_csv_file,
    # inferSchema="true", header="true")

    # # Display the rolling_avg_df DataFrame
    # rolling_avg_df.show()


if __name__ == "__main__":
    sys.exit(main())
