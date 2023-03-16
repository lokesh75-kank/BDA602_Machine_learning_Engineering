from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession

appName = "PySpark Example - MariaDB Example"
master = "local[*]"
# Create Spark session
spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

# database properties
database = "baseball"
user = "admin"
password = "1196"
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


class RollingAvgTransformer(
    Transformer,
    HasInputCols,
    HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    # def __init__(self, inputCols=None, outputCols=None):
    #     return

    def __init__(self, inputCols=None, outputCols=None, uid=None):
        super(RollingAvgTransformer, self).__init__()
        self._setDefault(inputCols=[], outputCols=[])

    def _transform(self, dataset):
        # input_cols = self.getInputCols()
        # output_cols = self.getOutputCols()
        # print(input_cols,"Input_cols")
        # print(output_cols,"output_col")

        # Reading sql through spark
        df_sql2_rolling_100days = spark.sql(
            """
           SELECT gdb.batter,gdb.game_id ,gdb.local_date,SUM(gdb.Hit)
           OVER ( PARTITION BY gdb.batter
           ORDER BY UNIX_TIMESTAMP (gdb.local_date)
           RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
           ) AS sum_Hit,
           SUM(gdb.atBat) OVER (
           PARTITION BY gdb.batter
           ORDER BY UNIX_TIMESTAMP (gdb.local_date)
           RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
           ) AS sum_atBat,
           SUM(1) OVER (
           PARTITION BY gdb.batter
           ORDER BY UNIX_TIMESTAMP(gdb.local_date)
           RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
           ) AS cnt
           FROM game_date_batter gdb
        """
        )

        df_sql2_rolling_100days.createOrReplaceTempView("rolling_100days")
        df_sql2_rolling_100days.persist(StorageLevel.DISK_ONLY)

        # create the sql3 for final output of rolling 100 avg
        final_result = spark.sql(
            """
               SELECT batter,game_id, local_date,sum_Hit,
               sum_atBat, CASE WHEN sum_atBat = 0 THEN 0 
               ELSE sum_Hit/sum_atBat END AS Batter_rolling_avg,
               cnt FROM rolling_100days WHERE sum_Hit IS NOT NULL
               ORDER BY batter,game_id, local_date
            """
        )
        return final_result
