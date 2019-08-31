# data-analysis
This is a general repository where I will store small data analysis projects, such as Jupyter Notebooks, or homework assignments.

## File Descriptions

### California Test Score Data Analysis.ipynb
This is a Jupyter Notebook using Python that conducts Principal Component Analysis and Linear Regression on a dataset describing California schools to try to predict their average test scores. The statsmodels and scikit-learn libraries are compared in the analysis.

### Data Mining Homework 1/2.r
As expected from the file names, these were two separate assignments in my fall 2018 data mining class written in R. The first conducts linear regression on a dataset of Toyota Corollas to predict their price. The second works with the same dataset, but also implements k-Nearest Neighbors regression and a regression decision tree. The price variable is also transformed to a binary High Price variable, which is used as the target variable for logistic regression, k-Nearest Neighbors classification and a classification decision tree.

### Tweet Collection/Tweet Processing
These files were written for my spring 2019 software development class. I collected tweets with the hashtags &#35;CaptainMarvel and &#35;WonderWoman over a period of several weeks in April and May in order to compare them and see if there were any differences in how people talked about them. In Tweet Collection, I scheduled a program to run every two hours that searched for tweets with these hashtags and stored them in a local MongoDB database. In Tweet Processing, basic data from the tweets is extracted into an Excel file, including a sentiment score, and the most common words and wordclouds are generated as well.

### Car Crash Analysis
This file was written for my spring 2019 capstone class. The project used the [NHTSA Traffic Fatalities dataset](https://www.kaggle.com/usdot/nhtsa-traffic-fatalities) as the basis for analysis. First the accidents tables were extracted from BigQuery. Then there was a first round of processing in order to produce a dataset suitable for visualizations. State population data is also gathered and included in this dataset. Then there is a second round of processing in order to produce a dataset suitable for machine learning algorithms. Finally, the classification decision tree and k-means clustering algorithms were run on the data.