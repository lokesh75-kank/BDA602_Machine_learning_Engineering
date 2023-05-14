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

## Feature Analysis

In the feature analysis for this project, the following tasks were performed:

1. Handled Outliers:
   - Outliers, which are data points that significantly deviate from the rest of the data, were identified.
   - Different techniques were used to handle outliers, such as removing them, replacing them with central tendency measures (e.g., median or mean), or capping them at a certain value.
   - Outliers were identified by comparing the values of a specific column (col) with lower and upper bounds.
   - If outliers were found, they were handled using either the median or mean method.
   - The np.clip function was used with the median method to limit outlier values to the lower and upper bounds.
   - With the mean method, a lambda function was applied to replace outlier values below the lower bound with the lower bound value and values above the upper bound with the upper bound value.
   
![outliers](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/dc89d83e-9330-4375-b163-73b5d91ad473)

2. Scaled Features using MinMaxScaler:
   - The values of features were scaled to a common scale using MinMaxScaler.
   - MinMaxScaler transformed the features to a specified range (e.g., between 0 and 1).
   - By applying MinMaxScaler to the features, they were brought to the same scale, improving the performance of certain machine learning algorithms that are sensitive to feature scale.


3. Visualized Data Distribution:
   - The distribution of data points was visualized to understand underlying patterns and characteristics.
   - Different types of plots were used based on the data types and the information being conveyed.
   - Commonly used plots, such as histograms, box plots, scatter plots, line plots, and bar plots, were employed.
   - Histograms provided insights into the distribution and frequency of data values.
   - Box plots displayed summary statistics (e.g., median, quartiles) and identified outliers in the data.
   - Scatter plots showed the relationship between two variables and helped identify patterns or correlations.
   - Line plots illustrated trends and changes over time.
   - Bar plots were used to visualize categorical data and compare different groups or categories.

e.g
![image](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/c31a5ceb-a327-4d5a-bf67-8fbabc229370)


![image](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/437603e3-0416-4f99-b364-c1730fa4df9d)


![image](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/b1ca9ac2-885f-4b87-bf22-65b0c990036d)


Outliers were handled, features were scaled using MinMaxScaler, and data distribution was visualized, enhancing the dataset for further analysis and modeling.

4. Calculated Weighted and Unweighted Metrics for Continuous and Categorical Features:
   - Continuous features were divided into bins to calculate weighted and unweighted metrics.
   - The number of bins was set to 10, and the range of each bin was determined based on the minimum and maximum values of the feature.
   - For each bin, the mean and count of the corresponding response variable were calculated.
   - The population mean of the response variable was also computed.
   - Weighted Mean Square Difference (MSD) was determined by multiplying the mean square difference of each bin by its population proportion.
   - Unweighted MSD was obtained by summing the mean square differences of all bins and dividing by the total number of bins.

5. Computed P and t Values for Continuous Predictors:
   - P and t values were calculated to assess the significance of continuous predictors in relation to the response variable.
   - Ordinary Least Squares (OLS) regression was performed using the continuous predictor and response variable.
   - The linear regression model was fitted, and the summary statistics were obtained.
   - The t value and p value of the predictor were extracted from the model summary.

e.g
## Feature Analysis Results

| Feature    | Plot       | Mean of Response Plot | Diff Mean Response (Weighted) | Diff Mean Response (Unweighted) | P-Value              | T-Score     |
|------------|------------|----------------------|-------------------------------|---------------------------------|----------------------|-------------|
| TB_Ratio   | TB_Ratio   | TB_Ratio             | 0.000328                      | 0.060422                        | 4.331031e-247        | 33.791093   |

- **Feature**: "TB_Ratio" represents the feature that was analyzed.
- **Plot**: "TB_Ratio" denotes the type of plot used to visualize the feature.
- **Mean of Response Plot**: "TB_Ratio" represents the plot that displays the mean response for different values of "TB_Ratio".
- **Diff Mean Response (Weighted)**: 0.000328 is the weighted difference in the mean response between different values of "TB_Ratio".
- **Diff Mean Response (Unweighted)**: 0.060422 is the unweighted difference in the mean response between different values of "TB_Ratio".
- **P-Value**: 4.331031e-247 is the p-value associated with the "TB_Ratio" feature.
- **T-Score**: 33.791093 represents the t-score for the "TB_Ratio" feature.

The provided result suggests that the "TB_Ratio" feature has a significant impact on the response variable. The small p-value and high t-score indicate strong evidence to reject the null hypothesis of no relationship between "TB_Ratio" and the response variable. The weighted and unweighted differences in the mean response further highlight the effect of the "TB_Ratio" feature on the response variable.



6.impurity-based feature importance analysis

The technique was applied to identify the relative importance of different features in predicting the target variable.

1. **Data Preparation:** The dataset was prepared by separating the target variable (home_team_wins) from the rest of the features (X) and storing it in the variable y.

2. **Model Training:** A Random Forest Regressor model with 100 estimators was utilized to capture the relationships between the features and the target variable. The model was trained using the prepared dataset (X and y).

3. **Computing Feature Importance:** The impurity-based feature importance was computed using the `RandomForestRegressor.feature_importances_` attribute. This attribute calculates the normalized total reduction in the criterion (e.g., mean squared error) brought by a feature across all trees in the random forest. Feature importances were computed for each feature in X.

4. **Printing Feature Importance:** The column names of the features and their corresponding importances were printed to the console. This information was obtained by pairing each feature name with its importance value using a loop.

The impurity-based feature importance analysis provides insights into the features that have the most significant impact on predicting the target variable (home_team_wins). This information can help prioritize and focus on the most influential features during subsequent modeling and analysis stages.




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
                                
                                  
                         
                         
