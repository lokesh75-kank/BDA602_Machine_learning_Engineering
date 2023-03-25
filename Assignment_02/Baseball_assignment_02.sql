
-- Annual batter average for each batter in each year they played.

-- CREATE TABLE annual_batting_avg AS 
SELECT bc.batter,YEAR(g.local_date) AS year, bc.game_id, bc.Hit ,bc.atBat, 
SUM(bc.hit)/NULLIF(SUM(bc.atbat),0) AS batting_average
FROM batter_counts bc 
JOIN game g ON g.game_id = bc.game_id
WHERE bc.Hit != 0 OR bc.atBat != 0
GROUP BY bc.batter, YEAR(g.local_date)
HAVING year IS NOT NULL
ORDER BY batter; 
 
--Explanation: 
--The SELECT statement selects the batter, the year of the game, the game_id, and the batting average.
-- The SUM(bc.hit) function calculates the total number of hits for each batter, while the SUM(bc.atBat) function calculates the total number of at-bats.
-- The WHERE clause filters out any rows where the batter did not get any hits or at-bats in a game.
-- The GROUP BY clause groups the results by the batter and year so that the batting average is calculated separately for each year a batter played in.
-- The HAVING clause filters out any rows where the year is null, which can happen if a game does not have a date associated with it.
-- The ORDER BY clause orders the results by batter name.
-- The NULLIF function is used in the calculation of the batting average to avoid division by zero errors.


-- Historic batting avg
SELECT bc.batter,bc.game_id,bc.Hit ,bc.atBat,SUM(bc.Hit) / NULLIF(SUM(bc.atBat), 0) AS batting_avg
FROM batter_counts bc
WHERE NOT bc.atBat = 0 
GROUP BY bc.batter
ORDER BY bc.batter;

--Explanation:
--The AVG function is used to calculate the batting average, which is the number of hits divided by the number of at-bats, and the NULLIF function is used to handle the case where the number of at-bats is zero. If the number of at-bats is zero, the NULLIF function returns NULL, which will cause the batting average to be NULL instead of causing a divide-by-zero error.
-- The GROUP BY clause groups the results by batter, so the AVG function is applied to each batter's hits and at-bats separately.


-- Add indexes to player_id column of both batter_counts and battersInGame tables
ALTER TABLE batter_counts ADD INDEX idx_batter (batter);
ALTER TABLE battersInGame ADD INDEX idx_batter (batter);

CREATE INDEX ix_batter_counts_game_id ON batter_counts(game_id);
CREATE INDEX ix_game_local_date ON game(local_date);

--Rolling average batting average of each batter for each game using the last 100 days of games played before the current game.


WITH batter_avg_temp_1 AS (
    SELECT bc.batter,bc.game_id, bc.Hit, bc.atBat,
    DATEDIFF(g.local_date, (SELECT MIN(local_date) FROM game)) AS days_diff
    FROM batter_counts bc
    LEFT JOIN game g ON g.game_id = bc.game_id
    WHERE atBat <> 0 ),    
temp_2 AS (
    SELECT batter, game_id, days_diff, SUM(Hit) / SUM(atBat) AS bat_avg
    FROM batter_avg_temp_1
    GROUP BY batter, game_id, days_diff )
    SELECT batter, game_id,
    (   SELECT AVG(bat_avg)
        FROM temp_2
        WHERE temp_2.days_diff BETWEEN t2.days_diff - 100 AND t2.days_diff - 1
        AND t2.batter = temp_2.batter
    ) AS batting_rolling_avg
FROM temp_2 t2  
WHERE (SELECT AVG(bat_avg)
        FROM temp_2
        WHERE temp_2.days_diff BETWEEN t2.days_diff - 100 AND t2.days_diff - 1
        AND t2.batter = temp_2.batter
    ) IS NOT NULL;
    
    
--Explanation: 
--The first part of the query (the batter_avg_temp_1 CTE) retrieves the necessary data for each batter's performance in each game, along with the number of days since the earliest game in the dataset.
-- The second part of the query (the temp_2 CTE) calculates the batting average for each batter in each game, using the SUM(Hit) / SUM(atBat) formula.
-- Finally, the outer query calculates the rolling average batting average for each batter in each game. It does this by using a correlated subquery that calculates the average of the batter's batting averages for the last 100 days of games played before the current game. The WHERE clause filters out any rows where the rolling average is null, since we don't want to include those in the final output.

-- corrected query
-- Create a "master" table

DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup;

CREATE TEMPORARY TABLE t_rolling_lookup AS
SELECT g.game_id ,DATE(local_date) AS local_date ,batter,atBat, Hit
FROM batter_counts bc
JOIN game g ON g.game_id = bc.game_id
ORDER BY batter, local_date;

CREATE UNIQUE INDEX rolling_lookup_date_game_batter_id_idx ON t_rolling_lookup (game_id, batter,
local_date);
CREATE UNIQUE INDEX rolling_lookup_game_batter_id_idx ON t_rolling_lookup (game_id, batter);
CREATE INDEX rolling_lookup_game_id_idx ON t_rolling_lookup (game_id); 
CREATE INDEX rolling_lookup_local_date_idx ON t_rolling_lookup (local_date); 
CREATE INDEX rolling_lookup_batter_idx ON t_rolling_lookup (batter);


CREATE TABLE rolling_100_partition_by AS
WITH subTable AS (
SELECT rl1.batter,rl1.game_id ,rl1.local_date,SUM(rl1.Hit) 
OVER ( 
PARTITION BY rl1.batter
ORDER BY UNIX_TIMESTAMP (rl1.local_date)
RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
) AS sum_Hit,
SUM(rl1.atBat) OVER (
PARTITION BY rl1.batter
ORDER BY UNIX_TIMESTAMP (rl1.local_date)
RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
) AS sum_atBat,
SUM(1) OVER (
PARTITION BY rl1.batter
ORDER BY UNIX_TIMESTAMP(rl1.local_date)
RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
) AS cnt
FROM t_rolling_lookup rl1
)

SELECT batter,game_id, local_date, sum_Hit,sum_atBat,
CASE
WHEN sum_atBat = 0 THEN 0 
ELSE sum_Hit/sum_atBat END AS BA, cnt
FROM subTable
WHERE sum_Hit IS NOT NULL
ORDER BY batter,game_id, local_date;
   
--changing alias for spark assignment:   

DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup;
sql1 = 
CREATE TEMPORARY TABLE game_date_batter_table AS
SELECT g.game_id ,DATE(local_date) AS local_date ,batter,atBat, Hit
FROM batter_counts bc
JOIN game g ON g.game_id = bc.game_id
ORDER BY batter, local_date;

CREATE UNIQUE INDEX game_date_batter_table_date_game_batter_id_idx ON game_date_batter_table (game_id, batter,
local_date);
CREATE UNIQUE INDEX game_date_batter_table_game_batter_id_idx ON game_date_batter_table (game_id, batter);
CREATE INDEX game_date_batter_table_game_id_idx ON game_date_batter_table (game_id); 
CREATE INDEX game_date_batter_table_local_date_idx ON game_date_batter_table (local_date); 
CREATE INDEX game_date_batter_table_batter_idx ON game_date_batter_table (batter);


sql2=
SELECT gdb.batter,gdb.game_id ,gdb.local_date,SUM(gdb.Hit) 
OVER ( 
PARTITION BY gdb.batter
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
FROM game_date_batter_ta



SELECT batter,game_id, local_date, sum_Hit,sum_atBat,
CASE
WHEN sum_atBat = 0 THEN 0 
ELSE sum_Hit/sum_atBat END AS BA, cnt
FROM subTable
WHERE sum_Hit IS NOT NULL
ORDER BY batter,game_id, local_date;
   
 
 
 
 
