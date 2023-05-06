CREATE TEMPORARY TABLE IF NOT EXISTS table_rolling_lookup AS
SELECT g.game_id, DATE(local_date) AS local_date, batter, atBat, Hit
FROM batter_counts bc
JOIN game g ON g.game_id = bc.game_id
ORDER BY batter, local_date;

CREATE UNIQUE INDEX IF NOT EXISTS rolling_lookup_date_game_batter_id_idx ON table_rolling_lookup (game_id, batter, local_date);
CREATE UNIQUE INDEX IF NOT EXISTS rolling_lookup_game_batter_id_idx ON table_rolling_lookup (game_id, batter);

DROP TABLE IF EXISTS rolling_100_partition;

CREATE TABLE IF NOT EXISTS rolling_100_partition AS
WITH subTable AS (
  SELECT rl1.batter, rl1.game_id, rl1.local_date, SUM(rl1.Hit)
  OVER (
    PARTITION BY rl1.batter
    ORDER BY UNIX_TIMESTAMP(rl1.local_date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS sum_Hit,
  SUM(rl1.atBat) OVER (
    PARTITION BY rl1.batter
    ORDER BY UNIX_TIMESTAMP(rl1.local_date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS sum_atBat,
  SUM(1) OVER (
    PARTITION BY rl1.batter
    ORDER BY UNIX_TIMESTAMP(rl1.local_date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS cnt
  FROM table_rolling_lookup rl1
)
SELECT batter, game_id, local_date, sum_Hit, sum_atBat,
  CASE
    WHEN sum_atBat = 0 THEN 0
    ELSE sum_Hit / sum_atBat
  END AS BA, cnt
FROM subTable
WHERE sum_Hit IS NOT NULL
ORDER BY batter, game_id, local_date;

CREATE INDEX IF NOT EXISTS rolling_lookup_game_id_idx ON table_rolling_lookup (game_id);
CREATE INDEX IF NOT EXISTS rolling_lookup_local_date_idx ON table_rolling_lookup (local_date);
CREATE INDEX IF NOT EXISTS rolling_lookup_batter_idx ON table_rolling_lookup (batter);

--SELECT * FROM rolling_100_partition WHERE game_id = 12560;
