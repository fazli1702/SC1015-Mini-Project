# US Accidents Analysis & Classification

**NTU SC1015 Mini Project**

## About
Each year, 1.35 million p.eople are killed on roadways around the world. Crash injuries are estimated to be the eighth leading cause of death globally for all age groups and the leading cause of death for children and young people 5–29 years of age.

Accidents are an unfortunate reality of our daily lives, impacting individuals, families, and communities across the United States. Understanding the causes, patterns, and consequences of accidents is of paramount importance for public safety, transportation planning, and policy-making. In this notebook, we delve into the world of accidents, specifically focusing on accidents that occur on the roads and highways of the United States.

## Contributors
| Name                         | Username                                     | Parts Done                                   |
|------------------------------|----------------------------------------------|----------------------------------------------|
| Mohamed Fazli Bin Mohd Yazid | [@fazli1702](https://github.com/fazli1702)   | Data Cleaning, EDA, Random Forest Classifier |
| Mohamed Faris Bin Mohd Yazid | [@faris1702](https://github.com/faris1702)   | EDA, Slides                                  |
| Muhammad Aidil Firdaus       | [@Aidilil](https://github.com/Aidilil)       | EDA (Geospatial Analysis), Neural Network    |

## Problem Statement
Given the different information surrounding an accident, we want to predict the `Severity` of an accident as accurately as possible. We also want to find which features influence `Severity` the most.

## Dataset
The dataset used for this project is located [here](https://drive.google.com/file/d/1U3u8QYzLjnEaSurtZfSAS_oh9AT2Mn8X/edit)  
The above dataset is sampled from the original dataset located [here](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

## Presentation
[Video]()  
[Slides]()

## Table of Content
1. Data Cleaning
2. Exploratory Data Analysis
3. Data Preperation
4. Machine Learning
5. Conclusion

## 1. [Data Cleaning](https://github.com/fazli1702/SC1015-Mini-Project/blob/main/Data_Cleaning.ipynb)
In this section of the project, we prepped and cleaned the dataset to help us analyze our data better.

We performed the following:
1. Sampled `30000` data rows from `California`, the state with the most accident to reduce the amount of data with will work with (>100000)
2. Drop columns that were deemed unnecessary / not useful for data analysis.
3. Standardisation of data
    - Convert `Start_Time` from `object` to `datetime` type
    - Convert numerical data with empirical units to metric units
    - Standardised all `zipcode` to the standard 5-digit and group them by first 3 digit i.e. `123XXX`
    - Convert the column `Sunrise_Sunset` with values `Day/Night` to the column `Sunlight` with boolean values `True/False`
    - Convert `Wind_Direction` from caridnal direction (e.g. `NW`/`SSE`) to degrees
4. Simplification
    - Simplified `Weather_Condition` from about 50 unique values down to 7 unique values (`Fair`, `Cloudy`, `Rainy`, `Snowy`, `Windy`, `Foggy`, `Others`) under the new column `Weather`
    - Simplified address `Street` according to road type. 5 Unique values (`Street`, `Avenue`, `Drive`, `Highway`, `Others`) under the new column `Road_Type`
5. Save cleaned data into csv file `csv/clean_data.csv`

## 2. [Exploratory Data Analysis](https://github.com/fazli1702/SC1015-Mini-Project/blob/main/Data_Exploration.ipynb)
With our cleaned data, we can now carry out exploratory data analysis. Here, we want to answer questions such as:
- Are there any patterns that we notice?
- What are the relationship between the variables and `Severity`
- Can we make any inferences for our question at this stage?

We performed the following:
1. **Severity**
    - We check that there are 4 `Severity` levels and `Severity = 2` is the most common, accounting for around 98% of all accidents
2. **Location of accidents**
    - We did geospatial analysis using `Start_Lng` and `Start_lat` and by looking at the map, we can clearly see that most accidents occurs within populated cities such as `Los Angeles` and `San Fransisco`. We also notice that a considerable amount of accidents occurs in roads connecting major cities
    - We further confirm our analysis by looking at `City`, `County` and `Zipcode`, which shows that most accidents occurs in major cities, with `Los Angeles` having the most accidents
    - From our analysis, the high number of accidents could be attributed to the high traffic within cities
3. **Weather**
    - We analyse the data using `Weather` and found similar trends across all 4 `Severity` levels, where `Fair` is the most common `Weather` condition, followed by `Cloudy`.
    - More adverse `Weather` condition such as `Foggy`, `Rainy` and `Snowy` have a very low accident count.
    - Our above observation could be attributed to the fact that there are lesser vehicle on the roads during adverse weather condition and those who are driving would be more careful and alert, leading to lesser accidents occuring compared to normal weather conditions. Drivers may also be less alert and careful during normal conditions and thus cause more accidents
4. **Road Type**
    - We analyse the data using `Road_Type` and found similar trends across all 4 `Severity` levels, where `Highway` is the most common, followed by `Street`.
    - Our observation could be attributed to the fact that `Highway` has a higher average speed and thus, more accidents are prone to occur
5. **Date & Time**
    - We analyse the data using `Start_Time`, which we further seperated into `Month`, `Day_Name` and `Hour` using the `Datetime` module.
    - `Month`: We notice accidents peaked at the end of the year and drops in the middle of the year
        - More people travelling during christmas period which is during winter causes peak
    - `Day_Name`: Most accidents occurs on the weekday
        - More people travel during weekday due to work
    - `Hour`: Peak in morning and evening period
        - Peak can be attributed to people travelling and coming back to work
    - From the above analysis, we created 4 new columns, `Is_Rush_Hour`, `Is_Christmas`, `Is_Weekday`, `Season`
6. **Season**
    - The column `Season`, which was created during our Date & Time analysis, shows that most accidents occurs during the `Winter`. However, this is only true for `Severity = 2`. The remaining `Severity` levels shows a similar trend where most accidents occurs during `Spring`
7. **Boolean Values**
    - We analyse variables with `bool` type, which shows the presence of **road fetures**, apart from the new `bool` columns that were added during our datetime analysis.
    - We found out that across all `Severity`, most accidents occurs when there is **NO** road features. This could be attributed to the fact that drivers may be more alert and careful when they see road features (e.g. `Stop`, `Giveway`signs, `Crossing` and `Bumb`) and thus lesser significantly much lesser accidents.
8. **Numerical Values**
    - We analyse variables with numerical type (e.g. `int64`, `float64`) such as `Visibility`, `Temperature` and `Precipitation`, which shows the current weather and road conditions.
    - We found out that most accidents occurs during normal road and weather conditions. This could be attributed to drivers being more careful during adverse road and weather conditions.
9. Save new data into csv file

## 3. [Data Preperation](https://github.com/fazli1702/SC1015-Mini-Project/blob/main/Data_Preparation.ipynb)
Having changed our dataframe slightly, we now need to prepare it for machine learning. We all columns into numerical data type to ease the process of machine learning later on

We performed the following:
1. Break `Start_Time` into `Year`, `Month`, `Day`, `Day_Week` & `Hour`, `Minute`
2. Remove unnecessary columns
3. Convert columns with `object` type into `int` type using `sklearn LabelEncoder`
4. Resampling
    - Since majority of our data is `Severity = 2`, we have an imbalanced dataset and thus need to do resampling
    - We used 3 different upsampling method
        1. SMOTE
        2. SMOTEENN
        3. SMOTETomek
5. Save prepared data into csv file

## 4. Machine Learning
We use 2 different machine learning methods in order to predict `Severity` based on the other variables
- [Random Forest Classification](https://github.com/fazli1702/SC1015-Mini-Project/blob/main/Random%20Forest.ipynb)
- [Neural Network](https://github.com/fazli1702/SC1015-Mini-Project/blob/main/neural-network-finalised.ipynb)

1. **Random Forest Classification**
    - Works by using multiple decision tree and learning from it, achieving better generalisation and robustness compared to individual decision trees
    - We encountered overfitting in our first attempt with `Train accuracy = 1.0 (perfect)` and `Test accuracy = 0.973`.
    - We tuned the hyperparameter of our Random Forest Classifier and achieve a more generalised accuracy with `Train accuracy = 0.77` and `Test accuracy = 0.797`
    - Feature importance: Top 5 most influential variables
        1. `Distance`
        2. `Month`
        3. `Start_Lng`
        4. `Start_Lat`
        5. `Zipcode`
    - The results are similar across all 3 different resampling methods

2. **Neural Network**
    - learn complex patterns and relationships in data through the iterative adjustment of weights and biases, guided by optimization algorithms during the training process

## 5. Conclusion
1. Drivers should pay more attention when roads are busy as most accidents occurs during this period
2. Installation of more road features can lead to a reduction in accidents
3. Drivers should not be complacent during good road and weather condition as most accidents occurs during this period

## What we learned
1. Geospatial analysis using `Folium` module
2. The different oversampling methods
    - SMOTE
    - SMOTEENN
    - SMOTETomek
3. The diffenent machine learning models
    - Random Forest
    - Neural Network

## References
- https://www.kaggle.com/code/muhammadaidilfirdaus/sc1015-geospatial-analysis#Mapping-Los-Angeles-Car-Accidents--Using-Folium-Heatmap
- https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.LabelEncoder.html
- https://stackoverflow.com/questions/44474570/sklearn-label-encoding-multiple-columns-pandas-dataframe
- https://towardsdatascience.com/statistical-learning-ii-data-sampling-resampling-93a0208d6bb8
- https://github.com/nicklimmm/movie-analysis/blob/main/data-resampling-and-splitting.ipynb
- https://www.datacamp.com/tutorial/random-forests-classifier-python
- https://www.geeksforgeeks.org/how-to-solve-overfitting-in-random-forest-in-python-sklearn/
- https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
- https://alexlenail.me/NN-SVG/index.html
