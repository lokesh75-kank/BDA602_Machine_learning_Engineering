![image](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/6ab579cf-edba-48db-b90e-b76d08c73b1c)

## Project Title

This project focuses on making prior forecasts of results in baseball games. By narrowing our focus to this binary classification, I aim to accurately predict whether a team will emerge as the winner or loser. Through data analysis and machine learning techniques,the goal is to enhance the accuracy of these predictions.

### Baseball

Baseball, though not as popular as cricket in India, has gained a niche following in recent years. The sport, introduced by American missionaries, has its unique terms that resonate with Indian fans. "Home run" signifies a powerful hit clearing the boundaries, while "strikeout" represents a batter being unable to make contact with the ball. "Pitcher" refers to the player delivering the ball, and "catcher" is the player behind the batter. Terms like "inning," "base," and "out" retain their universal meaning. As Indians explore baseball, these terminologies add a touch of familiarity, bridging the gap between cultures and infusing the game with local context.


<img src="https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/a8c76a17-9564-4897-98d9-7b6cc1b5eb2c" alt="maxresdefault" width="700" height="500">


## About baseball data

The baseball database, generously provided by [Prof. Julien Pierret](https://github.com/dafrenchyman), serves as a valuable resource for conducting in-depth analysis in the realm of baseball. Statistical data plays a pivotal role in enabling easy player comparisons and evaluating overall team performance on a given day. Leveraging this comprehensive database, which encompasses approximately 40 tables comprising pitching, hitting, and fielding statistics, I have extracted meaningful insights into both individual player performance and team dynamics. To access and work with the data, I utilized the mariaDB database management system and the Dbearver tool, facilitating efficient data exploration and analysis.
[Download Baseball SQL Data](https://www.teaching.mrsharky.com/data/baseball.sql.tar.gz)

### Feature Engineering
In this project, I have utilized various calculation methods to derive insightful features from the existing tables. These methods include comparing, contrasting, and quantifying the disparities between the home team and the away team's statistics. By employing these techniques, we gain a deeper understanding of the performance dynamics and inherent variations within the baseball data. These additional features enrich the analysis by providing nuanced perspectives on player and team capabilities, allowing for more comprehensive assessments and facilitating data-driven decision-making in the realm of baseball.

### List of Features Used

| Abbreviation              | Full Form                                |
|---------------------------|------------------------------------------|
| TB_Ratio                  | Total Bases Ratio                        |
| P_WHIP_Ratio              | Pitcher Walks plus Hits per Inning Ratio |
| HBP_Ratio                 | Hit by Pitch Ratio                       |
| A_HR_Ratio                | Away Team Home Run Ratio                 |
| ISO_Ratio                 | Isolated Power Ratio                     |
| BA_Ratio                  | Batting Average Ratio                    |
| SLG_Ratio                 | Slugging Percentage Ratio                |
| OBP_Ratio                 | On-Base Percentage Ratio                 |
| TOB_Ratio                 | Times on Base Ratio                      |
| P_ERA_Ratio               | Pitcher Earned Run Average Ratio         |
| HR9_Ratio                 | Home Runs per 9 Innings Ratio            |
| IP_Ratio                  | Innings Pitched Ratio                    |
| home_team_wins            | Home Team Wins                           |
| sum_atBat                 | Total At Bats                            |
| sum_Hit                   | Total Hits                               |
| sum_B                     | Total Bases                              |
| sum_2B                    | Total Doubles                            |
| sum_3B                    | Total Triples                            |
| sum_Home_Run              | Total Home Runs                          |
| sum_Sac_Fly               | Total Sacrifice Flies                    |
| sum_BB                    | Total Walks                              |
| sum_Fly_Out               | Total Fly Outs                           |
| sum_Hit_By_Pitch          | Total Hit By Pitch                       |
| sum_TB                    | Total Bases                              |
| sum_TOB                   | Total Times on Base                      |
| sum_Pitch_BB              | Total Pitcher Walks                      |
| sum_Pitch_IP              | Total Pitcher Innings Pitched            |
| sum_Pitch_Hit_By_Pitch    | Total Pitcher Hits by Pitch              |
| sum_Pitch_HR              | Total Pitcher Home Runs                  |
| avg_Pitch_AHR             | Average Pitcher Adjusted Home Runs       |
| sum_Pitch_K               | Total Pitcher Strikeouts                 |
| ERA                       | Earned Run Average                       |
| strikeout_per_inn         | Strikeouts per Inning                    |
| TotalBases_WHIP_Ratio     | Total Bases per Walks plus Hits Ratio    |
| Batting_Avg_SLG_Sum       | Batting Average plus Slugging Percentage |
| Times_On_Base_ERA_Difference  | Times on Base minus Earned Run Average Difference |
| HR_OBP_Ratio              | Home Run to On-Base Percentage Ratio     |
| HR9_IP_Product            | Home Runs per 9 Innings times Innings Pitched Product |
| WHIP_HBP_Ratio            | Walks plus Hits per Inning plus Hit by Pitch Ratio |
| TB_BA_Ratio               | Total Bases to Batting Average Ratio     |
| SLG_ERA_Difference        | Slugging Percentage minus Earned Run Average Difference |
| TOB_WHIP_Product          | Times on Base times Walks plus Hits per Inning Product |

This is the comprehensive list of features used in the project, encompassing a wide range of statistics and ratios. These features play a crucial role in the analysis and evaluation of player and team performance in the realm of baseball. For code please check out final_features.sql file under finals directory.


Installation
Explain the steps required to install and set up your project. Include any necessary dependencies and their versions, along with instructions on how to install them. If applicable, provide specific details on how to configure any environment variables or settings.

Usage
Describe how users can utilize your project. Provide clear instructions on how to run the code and any necessary input formats or parameters. If there are any examples or demonstrations, include them here. Additionally, you can include information on how to train the model or generate predictions.

Dataset
If your project uses a specific dataset, provide information about it in this section. Include details such as the source of the dataset, its format, and any preprocessing steps that were applied. If the dataset is publicly available, consider including a link to it.

Model Architecture
If your project involves a specific machine learning model or algorithm, describe its architecture or provide a high-level overview. Include any relevant equations, diagrams, or visualizations that can help readers understand the model's structure.

Results
Present the results of your machine learning project in this section. Include any performance metrics, accuracy scores, or visualizations that demonstrate the effectiveness of your model. If applicable, compare your results to previous approaches or state-of-the-art methods.

Future Work
Discuss potential future improvements or enhancements that could be made to your project. This could include additional features, alternative algorithms, or areas of research that could be explored further.

Contributing
If you want to encourage others to contribute to your project, provide guidelines on how they can do so. Explain the process for submitting bug reports, suggesting improvements, or proposing new features. Include information about your preferred coding style, branch structure, and how to submit pull requests.

License
Specify the license under which your project is distributed. If you're unsure about licensing, consult with your legal team or refer to popular open-source licenses like MIT, Apache, or GNU.

Acknowledgments
Use this section to thank anyone who contributed to the project, provided guidance, or inspired you. Acknowledge any relevant libraries, frameworks, or resources that you utilized.

Contact Information
Provide your contact information or any other means for users to reach out to you with questions or feedback. This can be your email address, GitHub profile, or any other preferred method of communication.

Remember, the above sections are just suggestions, and you can customize them according to the specific needs of your machine learning project.
                                
                                  
                         
                         
