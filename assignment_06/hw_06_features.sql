
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
tb_count.game_id,
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

 CREATE TEMPORARY TABLE IF NOT EXISTS temporary_features
 AS
 SELECT
 tb_count.*,
 b.away_runs,
 tr.home_streak,
 tr.away_streak,
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
 ON tb_count.game_id = tp_temp.game_id
 AND tb_count.team_id = tp_temp.team_id
 JOIN team_results tr
 ON tr.team_id = tb_count.team_id
 AND tr.game_id = tb_count.game_id
 JOIN boxscore b
 ON tb_count.game_id = b.game_id
 GROUP BY
 team_id,
 game_id;

CREATE TEMPORARY TABLE  100_rolling_days ENGINE=MEMORY AS
  SELECT
  ft1.team_id as team_id,
  ft1.game_id as game_id,
  NULLIF (SUM(ft2.atBat),0) AS atBat,
  NULLIF (SUM(ft2.Hit),0) AS Hit,
  NULLIF(SUM(ft2.Single),0) AS B,
  NULLIF(SUM(ft2.Double),0) AS 2B,
  NULLIF(SUM(ft2.Triple),0) AS 3B,
  NULLIF(SUM(ft2.Home_Run),0) AS Home_Run,
  NULLIF(SUM(ft2.Sac_Fly),0) AS Sac_Fly,
  NULLIF(SUM(ft2.Walk),0) AS BB,
  NULLIF(SUM(ft2.Fly_Out),0) AS Fly_Out,
  NULLIF(SUM(ft2.Hit_By_Pitch),0) AS Hit_By_Pitch,
  NULLIF(SUM(ft2.Single+2*ft2.Double+3*ft2.Triple+4*ft2.Home_Run),0) AS TB,
  NULLIF(SUM(ft2.Hit+ft2.Walk+ft2.Hit_By_Pitch),0) AS TOB,
  NULLIF(SUM(ft2.Pitch_Walk),0) AS Pitch_BB,
  NULLIF(SUM(ft2.Pitch_end_inn-ft2.Pitch_start_inn),0) as Pitch_IP,
  NULLIF(SUM(ft2.Pitch_Hit_by_Pitch),0) AS Pitch_Hit_By_Pitch,
  NULLIF(SUM(ft2.Pitch_Home_Run),0) AS Pitch_HR,
  NULLIF(AVG(ft2.Pitch_Home_Run),0) AS Pitch_AHR,
  NULLIF(SUM(ft2.Pitch_Strikeout),0) AS Pitch_K,
  NULLIF(9*(AVG(ft2.away_runs)/(ft2.Pitch_end_inn-ft2.Pitch_start_inn)),0) AS ERA
  FROM temporary_features ft1
  JOIN team t ON ft1.team_id = t.team_id
  JOIN game g1 ON g1.game_id = ft1.game_id
  JOIN temporary_features ft2 ON ft1.team_id = ft2.team_id
  JOIN game g2 ON g2.game_id = ft2.game_id AND g2.local_date < g1.local_date
  AND g2.local_date >= date_add(g1.local_date, INTERVAL - 100 day)
  GROUP BY ft1.team_id, ft1.game_id, g1.local_date
  ORDER BY ft1.team_id,g1.local_date;

--  CREATE UNIQUE INDEX 100_rolling_days_idx ON 100_rolling_days(team_id, game_id);

CREATE TABLE IF NOT EXISTS Baseball_features
SELECT g.game_id, g.home_team_id, g.away_team_id,
ROUND((rdh.TB / rda.TB),3) AS TB_Ratio,
ROUND((((rdh.Pitch_HR + rdh.Pitch_BB) / rdh.Pitch_IP) / ((rda.Pitch_HR + rda.Pitch_BB) / rda.Pitch_IP)),3) AS P_WHIP_Ratio,
ROUND((rdh.Hit_By_Pitch / NULLIF(rda.Hit_By_Pitch,0)),3) AS HBP_Ratio,
ROUND(((rdh.atBat/rdh.Home_Run)/(rda.atBat/rda.Home_Run)),3) AS A_HR_Ratio,
-- ROUND((((rdh.TB-rdh.B)/rdh.atBat)/NULLIF(((rda.TB-rda.B)/rda.atBat),0)),3) AS ISO_Ratio,
ROUND((((rdh.TB-rdh.B)/rdh.atBat)/NULLIF((rda.TB-rda.B),0)),3) AS ISO_Ratio,
-- ROUND(((rdh.Hit / rdh.atBat) / (rda.Hit / rda.atBat)),3) AS BA_Ratio,
CASE WHEN ROUND(((rdh.Hit / rdh.atBat) / (rda.Hit / rda.atBat)),5) < 0.5 THEN 0
     ELSE 1
     END AS BA_Ratio,
ROUND((((rdh.TB) / rdh.atBat) / ((rda.TB) / rda.atBat)),3) AS SLG_Ratio,
ROUND((((rdh.Hit + rdh.BB + rdh.Hit_By_Pitch) / (rdh.atBat + rdh.BB + rdh.Hit_By_Pitch + rdh.Sac_Fly))
 / ((rda.Hit + rda.BB + rda.Hit_By_Pitch) / (rda.atBat + rda.BB + rda.Hit_By_Pitch + rda.Sac_Fly))),3) AS OBP_Ratio,
ROUND((rdh.TOB / rda.TOB),3) AS TOB_Ratio,
ROUND((rdh.ERA / NULLIF(rda.ERA,0)),3) AS P_ERA_Ratio,
ROUND(((rdh.Pitch_AHR)/rdh.Pitch_IP)/NULLIF(((rda.Pitch_AHR)/rda.Pitch_IP),0),3) AS HR9_Ratio,
ROUND((rdh.Pitch_IP/rda.Pitch_IP),3) AS IP_Ratio,
CASE WHEN b.away_runs < b.home_runs THEN 1
     WHEN b.away_runs > b.home_runs THEN 0
     ELSE 0 END AS home_team_wins FROM
game g JOIN 100_rolling_days rdh
ON g.game_id = rdh.game_id AND g.home_team_id = rdh.team_id
JOIN 100_rolling_days rda ON g.game_id = rda.game_id AND g.away_team_id = rda.team_id
JOIN boxscore b ON b.game_id = g.game_id;