
DROP TEMPORARY TABLE IF EXISTS tempory_team_pitching;

CREATE TEMPORARY TABLE IF NOT EXISTS tempory_team_pitching
ENGINE = MEMORY AS SELECT
tb_count.win,
tb_count.atBat,
tb_count.Hit,
tb_count.Fly_Out,
tb_count.Hit_By_Pitch,
tb_count.Home_Run,
tb_count.Strikeout,
tb_count.Walk,
tb_count.game_id as pitch_game_id,
tb_count.team_id,
p_count.startingPitcher,
SUM(p_count.startingInning) AS start_inn,
SUM(p_count.endingInning) AS end_inn
FROM team_pitching_counts tb_count JOIN pitcher_counts p_count
ON p_count.game_id = tb_count.game_id
AND p_count.team_id = tb_count.team_id
GROUP BY p_count.team_id, p_count.game_id;


ALTER TABLE tempory_team_pitching
CHANGE COLUMN win Pitch_win INT NULL DEFAULT 0,
CHANGE COLUMN atBat Pitch_atBat INT NULL DEFAULT 0,
CHANGE COLUMN Hit Pitch_Hit INT NULL DEFAULT 0,
CHANGE COLUMN Fly_Out Pitch_Fly_Out FLOAT NULL DEFAULT 0,
CHANGE COLUMN Hit_By_Pitch Pitch_Hit_By_Pitch FLOAT NULL DEFAULT 0,
CHANGE COLUMN Home_Run Pitch_Home_Run FLOAT NULL DEFAULT 0,
CHANGE COLUMN Strikeout Pitch_Strikeout FLOAT NULL DEFAULT 0,
CHANGE COLUMN Walk Pitch_Walk FLOAT NULL DEFAULT 0,
CHANGE COLUMN start_inn Pitch_start_inn DOUBLE NULL DEFAULT 0,
CHANGE COLUMN end_inn Pitch_end_inn DOUBLE NULL DEFAULT 0;


DROP TEMPORARY TABLE IF EXISTS temporary_features;
CREATE TEMPORARY TABLE IF NOT EXISTS temporary_features
AS
SELECT
tb_count.*,
b.away_runs,
tp_temp.Pitch_win,
tp_temp.Pitch_atBat,
tp_temp.Pitch_Hit,
tp_temp.Pitch_Hit_By_Pitch,
tp_temp.Pitch_Home_Run,
tp_temp.Pitch_Strikeout,
tp_temp.Pitch_Walk,
tp_temp.Pitch_start_inn,
tp_temp.Pitch_end_inn
FROM
team_batting_counts tb_count
JOIN tempory_team_pitching tp_temp
ON tb_count.game_id = tp_temp.pitch_game_id
AND tb_count.team_id = tp_temp.team_id
JOIN team_results tr
ON tr.team_id = tb_count.team_id
AND tr.game_id = tb_count.game_id
JOIN boxscore b
ON tb_count.game_id = b.game_id
GROUP BY
team_id,
game_id;



DROP TEMPORARY TABLE IF EXISTS Baseball_rolling_100_stats;


CREATE TEMPORARY TABLE IF NOT EXISTS Baseball_rolling_100_stats ENGINE=MEMORY AS
SELECT
ft1.team_id as common_team_id,
ft1.game_id as common_game_id,
NULLIF (SUM(ft2.atBat),0) AS sum_atBat,
NULLIF (SUM(ft2.Hit),0) AS sum_Hit,
NULLIF(SUM(ft2.Single),0) AS sum_B,
NULLIF(SUM(ft2.Double),0) AS sum_2B,
NULLIF(SUM(ft2.Triple),0) AS sum_3B,
NULLIF(SUM(ft2.Home_Run),0) AS sum_Home_Run,
NULLIF(SUM(ft2.Sac_Fly),0) AS sum_Sac_Fly,
NULLIF(SUM(ft2.Walk),0) AS sum_BB,
NULLIF(SUM(ft2.Fly_Out),0) AS sum_Fly_Out,
NULLIF(SUM(ft2.Hit_By_Pitch),0) AS sum_Hit_By_Pitch,
NULLIF(SUM(ft2.Single+2*ft2.Double+3*ft2.Triple+4*ft2.Home_Run),0) AS sum_TB,
NULLIF(SUM(ft2.Hit+ft2.Walk+ft2.Hit_By_Pitch),0) AS sum_TOB,
NULLIF(SUM(ft2.Pitch_Walk),0) AS sum_Pitch_BB,
NULLIF(SUM(ft2.Pitch_end_inn-ft2.Pitch_start_inn),0) as sum_Pitch_IP,
NULLIF(SUM(ft2.Pitch_Hit_by_Pitch),0) AS sum_Pitch_Hit_By_Pitch,
NULLIF(SUM(ft2.Pitch_Home_Run),0) AS sum_Pitch_HR,
NULLIF(AVG(ft2.Pitch_Home_Run),0) AS avg_Pitch_AHR,
NULLIF(SUM(ft2.Pitch_Strikeout),0) AS sum_Pitch_K,
NULLIF(9*(AVG(ft2.away_runs)/(ft2.Pitch_end_inn-ft2.Pitch_start_inn)),0) AS ERA
FROM temporary_features ft1
JOIN team t ON ft1.team_id = t.team_id
JOIN game g1 ON g1.game_id = ft1.game_id
JOIN temporary_features ft2 ON ft1.team_id = ft2.team_id
JOIN game g2 ON g2.game_id = ft2.game_id AND g2.local_date < g1.local_date
AND g2.local_date >= date_add(g1.local_date, INTERVAL - 100 day)
GROUP BY ft1.team_id, ft1.game_id, g1.local_date
ORDER BY ft1.team_id,g1.local_date;

-- Create an index for the Baseball_rolling_100_stats table
CREATE INDEX idx_common_game_id ON Baseball_rolling_100_stats (common_game_id);

DROP TEMPORARY TABLE IF EXISTS Baseball_ratio_features;

CREATE TEMPORARY TABLE IF NOT EXISTS Baseball_ratio_features
SELECT g.game_id as game_id_ratio, g.home_team_id as home_team_id_ratio, g.away_team_id,
ROUND((bhd.sum_TB / bhd2.sum_TB),3) AS TB_Ratio,
ROUND((((bhd.sum_Pitch_HR + bhd.sum_Pitch_BB) / bhd.sum_Pitch_IP) / ((bhd2.sum_Pitch_HR + bhd2.sum_Pitch_BB) / bhd2.sum_Pitch_IP)),3) AS P_WHIP_Ratio,
ROUND((bhd.sum_Hit_By_Pitch / NULLIF(bhd2.sum_Hit_By_Pitch,0)),3) AS HBP_Ratio,
ROUND(((bhd.sum_atBat/bhd.sum_Home_Run)/(bhd2.sum_atBat/bhd2.sum_Home_Run)),3) AS A_HR_Ratio,
-- ROUND((((bhd.TB-bhd.B)/bhd.atBat)/NULLIF(((bhd2.TB-bhd2.B)/bhd2.atBat),0)),3) AS ISO_Ratio,
ROUND((((bhd.sum_TB-bhd.sum_B)/bhd.sum_atBat)/NULLIF((bhd2.sum_TB-bhd2.sum_B),0)),3) AS ISO_Ratio,
-- ROUND(((bhd.Hit / bhd.atBat) / (bhd2.Hit / bhd2.atBat)),3) AS BA_Ratio,
CASE WHEN ROUND(((bhd.sum_Hit / bhd.sum_atBat) / (bhd2.sum_Hit / bhd2.sum_atBat)),5) < 0.5 THEN 0
ELSE 1
END AS BA_Ratio,
ROUND((((bhd.sum_TB) / bhd.sum_atBat) / ((bhd2.sum_TB) / bhd2.sum_atBat)),3) AS SLG_Ratio,
ROUND((((bhd.sum_Hit + bhd.sum_BB + bhd.sum_Hit_By_Pitch) / (bhd.sum_atBat + bhd.sum_BB + bhd.sum_Hit_By_Pitch + bhd.sum_Sac_Fly))
/ ((bhd2.sum_Hit + bhd2.sum_BB + bhd2.sum_Hit_By_Pitch) / (bhd2.sum_atBat + bhd2.sum_BB + bhd2.sum_Hit_By_Pitch + bhd2.sum_Sac_Fly))),3) AS OBP_Ratio,
ROUND((bhd.sum_TOB / bhd2.sum_TOB),3) AS TOB_Ratio,
ROUND((bhd.ERA / NULLIF(bhd2.ERA,0)),3) AS P_ERA_Ratio,
ROUND(((bhd.avg_Pitch_AHR)/bhd.sum_Pitch_IP)/NULLIF(((bhd2.avg_Pitch_AHR)/bhd2.sum_Pitch_IP),0),3) AS HR9_Ratio,
ROUND((bhd.sum_Pitch_IP/bhd2.sum_Pitch_IP),3) AS IP_Ratio,
CASE WHEN b.away_runs < b.home_runs THEN 1
WHEN b.away_runs > b.home_runs THEN 0
ELSE 0 END AS home_team_wins FROM
game g JOIN Baseball_rolling_100_stats bhd
ON g.game_id = bhd.common_game_id AND g.home_team_id = bhd.common_team_id
JOIN Baseball_rolling_100_stats bhd2 ON g.game_id = bhd2.common_game_id AND g.away_team_id = bhd2.common_team_id
JOIN boxscore b ON b.game_id = g.game_id;

CREATE INDEX idx_game_id_ratio ON Baseball_ratio_features (game_id_ratio);

DROP TABLE IF EXISTS per_game_features;

CREATE TEMPORARY TABLE IF NOT EXISTS per_game_features AS
SELECT bf.*, bhd.*,
ROUND((((9*(bhd.sum_Pitch_K/bhd.sum_Pitch_IP))-(9*(bhd2.sum_Pitch_K/bhd2.sum_Pitch_IP)))/(9*(bhd.sum_Pitch_K/bhd.sum_Pitch_IP)))*100,3) AS strikout_per_inn
FROM Baseball_ratio_features bf
JOIN Baseball_rolling_100_stats bhd ON bf.game_id_ratio = bhd.common_game_id
JOIN Baseball_rolling_100_stats bhd2 ON bf.game_id_ratio = bhd2.common_game_id;

DROP TEMPORARY TABLE IF EXISTS game_year;
CREATE TEMPORARY TABLE IF NOT EXISTS game_year AS
SELECT g.game_id AS bag_game_id, YEAR(g.local_date) AS year
FROM game g
GROUP BY g.game_id, YEAR(g.local_date);


DROP TABLE IF EXISTS batting_common_features;
CREATE TEMPORARY TABLE IF NOT EXISTS batting_common_features
select * from game_year gy join per_game_features crf
on gy.bag_game_id = crf.game_id_ratio  ;

DROP TABLE IF EXISTS baseball_final_features;

CREATE TABLE IF NOT EXISTS baseball_final_features AS
SELECT
bcf.*,
(bf.TB_Ratio * bf.P_WHIP_Ratio) AS TotalBases_WHIP_Ratio,
(bf.HBP_Ratio / NULLIF(bf.ISO_Ratio, 0)) AS HBP_ISO_Ratio,
(bf.BA_Ratio + bf.SLG_Ratio) AS Batting_Avg_SLG_Sum,
(bf.TOB_Ratio - bf.P_ERA_Ratio) AS Times_On_Base_ERA_Difference,
(bf.A_HR_Ratio / NULLIF(bf.OBP_Ratio, 0)) AS HR_OBP_Ratio,
(bf.HR9_Ratio * bf.IP_Ratio) AS HR9_IP_Product,
(bf.P_WHIP_Ratio / NULLIF(bf.HBP_Ratio, 0)) AS WHIP_HBP_Ratio,
(bf.TB_Ratio / NULLIF(bf.BA_Ratio, 0)) AS TB_BA_Ratio,
(bf.SLG_Ratio - bf.P_ERA_Ratio) AS SLG_ERA_Difference,
(bf.TOB_Ratio * bf.P_WHIP_Ratio) AS TOB_WHIP_Product
FROM batting_common_features bcf
JOIN Baseball_ratio_features bf ON bcf.bag_game_id = bf.game_id_ratio;

-- Create an index for the baseball_final_features table
CREATE INDEX idx_game_id_ratio ON baseball_final_features (bag_game_id);
--SELECT * from baseball_final_features;
