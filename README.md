# german-credit-risk
This project is using the modified version of German credit data-set 
from UCI machine learning repository to classify loan applicants as a good or bad credit risk.


# Objective
The German credit data represented by 20 decision attributes and a class attribute. 
Each case is listed and classified as ‘Good credit risk’ or ‘Bad credit risk’ 
which is encoded as Class labels ‘1’ and ‘2’
respectively. Various predictive models were built and validated as part 
of this project classify loan applicants as a good or bad credit risk. Some of the 
classifiers used were Naive Bayes, Random Forest, and Neural Network. 
Plus, ensemble methods like Bagging and Boosting were also used to
discover their effect on the model's performance. 



# Dataset
GCD (German Credit Data) data set, includes 21 different attributes of each instances.
"1" indicates good credit risk and "2" indicates bad credit risk.
Excluding the Class attribute(dependent variable, which is the most 
crucial attribute forour analyses), there are 7 numerical 
attributes and in total of 13 categorical attributes.



# Code Information
The data frame is split into 2 data frames with the ratio of 70:30. The bigger data
frame(70% of our main data frame) is to train the classification and the smaller one(30%
of our data set) is to test our model. The Train set is generated randomly and the test
set is the complement of our train set.
There are 5 different models (Decision Tree, Naive Bayes, Bagging, Boosting, and
Random Forest) and were created to train our model and test it. Class attribute is the
response variable and all the other attributes are the predictors.



