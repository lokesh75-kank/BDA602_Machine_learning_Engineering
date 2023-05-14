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


| Feature                      | Importance               |
|------------------------------|--------------------------|
| IP_Ratio                     | 0.0503                   |
| WHIP_HBP_Ratio               | 0.0502                   |
| P_WHIP_Ratio                 | 0.0461                   |
| HR_OBP_Ratio                 | 0.0463                   |
| TOB_Ratio                    | 0.0445                   |
| A_HR_Ratio                   | 0.0443                   |
| OBP_Ratio                    | 0.0450                   |
| TotalBases_WHIP_Ratio        | 0.0443                   |
| HR9_Ratio                    | 0.0437                   |
| HR9_IP_Product               | 0.0437                   |
| ISO_Ratio                    | 0.0010                   |
| BA_Ratio                     | 0.0001                   |
| strikout_per_inn             | 0.0000                   |
| scaled_strikout_per_inn      | 0.0000                   |


The table above shows the feature importance values for the analyzed features. The top features with the highest importance values are IP_Ratio, WHIP_HBP_Ratio, and P_WHIP_Ratio, indicating their strong influence on the target variable. Conversely, features like ISO_Ratio, BA_Ratio, and strikout_per_inn have negligible importance in the analysis.


7. Difference with mean of response

In the feature analysis, the weighted mean squared difference was calculated for each feature using 10 bins. This involved comparing the average response value in each bin to the population average response. A higher rank in mean squared difference indicates a better predictor, as it signifies a larger difference between the average response values in the bins and the population average response.

 e.g
 sum_BB
 ![image](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/1241a538-4ba9-41eb-8c29-8bb71dace42e)
sum_Sac_Fly
![image](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/842f1be8-5806-481c-bdd8-f7f8e57d5f74)

       
 8. Correlation Matrix
 
The Pearson correlation coefficient method was employed to determine the correlation between different types of features. This method measures the linear relationship between two variables by calculating their covariance and dividing it by the product of their standard deviations.

Firstly, categorical-categorical features were evaluated to assess the association between two categorical variables. The correlation coefficient ranges from -1 to 1, where values closer to 1 indicate a strong positive relationship, values closer to -1 represent a strong negative relationship, and values close to 0 imply no significant correlation.

Secondly, categorical-continuous features were analyzed to examine the connection between a categorical variable and a continuous variable. The correlation coefficient provides insights into the strength and direction of the relationship.

Lastly, continuous-continuous features were assessed to determine the linear correlation between two continuous variables. The correlation coefficient helps understand the degree of dependence between the variables, with values near 1 or -1 indicating a strong positive or negative linear relationship, respectively.

By employing the Pearson correlation coefficient method, valuable insights were obtained regarding the interdependencies among different types of features in the dataset.

![image](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/98b524ee-b369-4633-b230-0b1c8e65d44f)

In this plot we can see some features are hightly correlated so we can remove either of them. for e.g Times_On_Base_ERA_Difference and SLG_ERA_Difference

## Model Building

### Handed Null Values

Null values were filled with 0 to preserve the structure and integrity of the dataset, especially for numerical features, and to avoid data loss.

### Sorted the DataFrame

The DataFrame was sorted based on the "year" column to ensure chronological order.

### Determined the Cutoff Year

The maximum year in the sorted DataFrame was subtracted by 1 to determine the cutoff year for splitting the data.

### Splitted the Data

The DataFrame was split into two parts: X (features) and y (target variable). The index where the cutoff year starts was identified, and the training and testing sets were created based on the cutoff index.

### Reset Index

The index of the training and testing sets for X and y were reset to ensure sequential indexing.


### Models Used
I implemented the following models for analysis:

- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors (KNN) Model
- Gradient Boosting Model
- XGBoost Model
- Decision Tree Model

### Evaluation Techniques
To assess the performance of the models, I employed various evaluation techniques, including:

- ROC Curve and AUC Score
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- 
By utilizing these evaluation techniques, I gained insights into the models' predictive abilities and their performance across different metrics.
 
## Logistic Regression
| Model                 | Accuracy   |
|-----------------------|------------|
| Logistic Regression   | 0.548      |

Classification Report of Logistic Regression:

|           | precision  | recall  | f1-score | support |
|-----------|------------|---------|----------|---------|
|    0      |   0.51     |  0.08   |   0.14   |   6592  |
|    1      |   0.55     |  0.94   |   0.69   |   7944  |

|   accuracy              |    0.55    |
|-------------------------|------------|
|   macro avg             |    0.53    |
|   weighted avg          |    0.53    |

| Model               | Accuracy | Precision | Recall | F1-score | AUC    |
|---------------------|----------|-----------|--------|----------|--------|
| Logistic Regression | 0.5484   | 0.5511    | 0.9373 | 0.6941   | 0.5462 |

![Logistic_Regression_ROC](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/2bddc0dc-7bca-4482-ba8e-15a0a6db178b)

![Logistic_Regression_Confusion_Matrix](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/0587e1ef-04e9-4056-b99c-41f072927d59)

## Random_Forest

| Model              | Accuracy |
|--------------------|----------|
| Random Forest      | 0.5116   |

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.45      | 0.34   | 0.39     | 6592    |
| Class 1   | 0.54      | 0.66   | 0.59     | 7944    |
|-----------|-----------|--------|----------|---------|
| accuracy  |           |        | 0.51     | 14536   |
| macro avg | 0.50      | 0.50   | 0.49     | 14536   |
| weighted avg | 0.50   | 0.51   | 0.50     | 14536   |

      
|      Model       |  Accuracy  | Precision |  Recall   | F1-score  |    AUC    |
|----------------- |------------|-----------|-----------|-----------|-----------|
| Random Forest    | 0.511557   | 0.544050  | 0.656093  | 0.594841  | 0.498470  |


![Random_Forest_ROC](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/2cf12494-b982-4213-b353-a0b5969c38da)

![Random_Forest_Confusion_Matrix](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/a5c89c36-9866-474b-934a-a8951dff88ed)

## KNN

| Metric                        | Value     |
|-------------------------------|-----------|
| Accuracy of KNN               | 0.5218767 |
|                               |           |
| Classification Report         |           |
|                               |           |
|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.46      | 0.36   | 0.40     | 6592    |
| 1             | 0.55      | 0.66   | 0.60     | 7944    |
|               |           |        |          |         |
| accuracy      |           |        |          | 0.52    |
| macro avg     | 0.51      | 0.51   | 0.50     | 14536   |
| weighted avg  | 0.51      | 0.52   | 0.51     | 14536   |

|     Model    |    Accuracy   |   Precision   |    Recall    |   F1-score   |     AUC      |
|--------------|---------------|---------------|--------------|--------------|--------------|
|     KNN      |    0.5219     |    0.5526     |    0.6576    |    0.6005    |    0.5109    |


![KNN_ROC](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/916d8741-1eab-4acf-9a1f-fbd565462c24)

![KNN_Confusion_Matrix](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/e31e0781-ccb4-41e5-8fe2-ff312d9d5434)


## Gradian_Boosting

|                           | precision | recall | f1-score | support |
|---------------------------|-----------|--------|----------|---------|
|           0               |   0.48    |  0.25  |   0.33   |   6592  |
|           1               |   0.56    |  0.78  |   0.65   |   7944  |
|---------------------------|-----------|--------|----------|---------|
|      accuracy             |           |        |   0.54   |  14536  |
|      macro avg            |   0.52    |  0.51  |   0.49   |  14536  |
|   weighted avg            |   0.52    |  0.54  |   0.50   |  14536  |

|     Model       |   Accuracy   |  Precision  |    Recall   |  F1-score   |     AUC     |
|-----------------|--------------|-------------|-------------|-------------|-------------|
| Gradient Boosting|   0.538      |   0.555     |   0.775     |   0.647     |   0.520     |

![Gradient_Boosting_ROC](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/b2577e8e-1dea-4982-a35a-3394d1906f06)
![Gradient_Boosting_Confusion_Matrix](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/c26911a7-0d9f-4e42-a39e-b33c7bee63ea)


## XGBoost

| Metric              | Value      |
|---------------------|------------|
| Accuracy            | 0.5289     |
| Precision (class 0) | 0.47       |
| Precision (class 1) | 0.55       |
| Recall (class 0)    | 0.31       |
| Recall (class 1)    | 0.71       |
| F1-score (class 0)  | 0.37       |
| F1-score (class 1)  | 0.62       |
| Support (class 0)   | 6592       |
| Support (class 1)   | 7944       |
| Macro Avg (Precision, Recall, F1-score) | 0.51 |
| Weighted Avg (Precision, Recall, F1-score) | 0.52 |
| Accuracy (overall)  | 0.53       |
| Total support       | 14536      |

| Metric    | Value                |
|-----------|----------------------|
| Accuracy  | 0.5288937809576224  |
| Precision | 0.5536203522504892  |
| Recall    | 0.7122356495468278  |
| F1-score  | 0.6229905307201057  |
| AUC       | 0.5075463392411932  |

![XGBoost_ROC](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/758df80b-56cc-4d2a-84af-c2841d43f49a)

![XGBoost_Confusion_Matrix](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/b8142936-788f-4d71-9e53-79fb8db1f0d8)

## Decision Tree

| Metric                             | Value     |
|------------------------------------|-----------|
| Accuracy of Decision Tree           | 0.5151348 |
|                                    |           |
| Classification Report of Decision  |           |
| Tree Classifier                    |           |
| Precision (0)                      | 0.47      |
| Recall (0)                         | 0.47      |
| F1-score (0)                       | 0.47      |
| Support (0)                        | 6592      |
|                                    |           |
| Precision (1)                      | 0.56      |
| Recall (1)                         | 0.55      |
| F1-score (1)                       | 0.55      |
| Support (1)                        | 7944      |
|                                    |           |
| Accuracy                           | 0.52      |
| Macro Average (f1-score)           | 0.51      |
| Macro Average (weighted avg)       | 0.51      |
| Weighted Average (f1-score)        | 0.52      |
| Weighted Average (weighted avg)    | 0.52      |
| Support                            | 14536     |


|     Model    |    Accuracy    |   Precision   |    Recall     |   F1-score    |      AUC      |
|:------------:|:--------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Decision Tree|   0.5151348   |   0.5569975   |   0.5511078   |   0.5540369   |   0.5114459   |

![Decision_Tree_ROC](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/ffb13b9e-9ea0-4bfc-846c-971805f4a562)
![Decision_Tree_Confusion_Matrix](https://github.com/lokesh75-kank/BDA602_Machine_learning_Engineering/assets/85188079/407483fa-b80e-4c34-9ed3-f4c997ebe3e1)

## After comparing the models, here's a brief summary of the results:

Logistic Regression has the highest accuracy (0.546) and a relatively high recall (0.962),
indicating that it correctly identifies a large proportion of positive instances.
Random Forest has lower accuracy (0.500) but shows balanced precision (0.535) and
recall (0.646) scores.
KNN has moderate accuracy (0.523) and performs reasonably well in terms of precision
(0.553) and recall (0.659).
Gradient Boosting has a relatively high recall (0.812), indicating it identifies a
higher proportion of positive instances, but has lower accuracy (0.541).
XGBoost has similar performance to the other models with moderate accuracy (0.520)
and precision (0.552) scores.
Decision Tree has the lowest accuracy (0.505) and relatively low precision (0.548)
and recall (0.540) scores.
Based on these results, the Logistic Regression model appears to be the best overall,
as it has the highest accuracy and reasonably balanced precision and recall scores.

## Docker and Run this project

To run this project, please follow the steps below:

1. **Install MariaDB:** Ensure that you have Mariadb installed on your system.

2. **Install Docker Engine and Docker Compose:** Make sure you have Docker Engine and Docker Compose installed. These tools are required to build and run the Docker containers for this project.

3. **Download the database file:** Download the necessary database file from the link provided in the project report. This file contains the required data for the project.

4. **Clone the final GitHub repository:** Clone the final GitHub repository for this project onto your local machine. This repository contains all the code and configuration files needed to run the project.

5. **Navigate to the project directory:** Using the command line, navigate to the directory where you cloned the GitHub repository.

6. **Build and run the Docker containers:** Run the `docker-compose up` command in the project directory. This command will build and run the Docker containers, which include all the necessary dependencies and configurations for the project.

7. **Check the results and generated plots:** After the Docker containers have started, you can find the results and generated plots in the directory. This directory will contain the output files and visualizations produced by the project.

Please make sure to follow these steps in order to successfully run the project and access the results.


## Reference

- https://en.wikipedia.org/wiki/Baseball_statistics
- https://www.teaching.mrsharky.com/index.html
- https://en.wikipedia.org/wiki/Variance_of_the_mean_and_predicted_responses
- https://docs.docker.com/reference/
- https://mariadb.com/kb/en/documentation/
