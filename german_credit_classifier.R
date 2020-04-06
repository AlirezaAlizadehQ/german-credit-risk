rm(list=ls())
options(digits = 4)

#installing packages
install.packages('tree')
library(tree)
install.packages('e1071')
library(e1071)
install.packages(('ROCR'))
library(ROCR)
install.packages('rpart')
library(rpart)
install.packages('adabag')
library(adabag)
install.packages('randomForest')
library(randomForest)
install.packages('neuralnet')
library(neuralnet)

gcd <- read.csv("GCD2018.csv")
gcd2018 = read.csv("GCD2018.csv")


# to show the structure the data set
str(gcd)

# to show the suummary of data ftame
summary(gcd)

# make a table for class column to examine the ratio of 1:2 
# to evaluate the ratio of goodCreditRisk:badCreditRisk
gcd.table = table(gcd$Class)
gcd.table.prop = prop.table(gcd.table)
gcd.table.prop
# the ratio of good:bad is 0.69999:0.3001

# convert ratio to percentage
gcd.table.prop.perc = gcd.table.prop * 100
gcd.table.prop.perc
# good:69.99%   bad:30.01% 

# cleaning the data 
# in some fileds there are missing values, which indicate with NA
# we must either eliminate those values, or substitute them 
# with the proper value such ad mode or mean of the table

# use average age for NA fields for age
avg.age = mean(gcd$Age, na.rm = TRUE)
gcd$Age[is.na(gcd$Age)] = floor(avg.age)

nrow(gcd)
# 1000 rows

# clean those fileds where they have NA values
gcd = na.omit(gcd)

# gcd2018 is same as gcd, but does not use mean for 
# NA fields in age attribute
gcd2018 = na.omit(gcd2018)

# to ckeck if NA fields were removed from Class attribute
is.na(gcd$Class)

nrow(gcd)
# 817 rows. 
# 183 row were removed.

nrow(gcd2018)
# 812 rows
# 188 rows were eliminated

# to convert the Class attribute to a factor
gcd.factor.class = factor(gcd$Class)
gcd.factor.class
gcd.clean.factor = cbind(gcd,gcd.factor.class)
str(gcd.clean.factor)
gcd.clean.factor$Class = NULL
colnames(gcd.clean.factor)[21] = 'Class'
str(gcd.clean.factor)
summary(gcd.clean.factor)

gcd2018.factor.class = factor(gcd2018$Class)
gcd2018.factor.class
gcd2018.clean.factor = cbind(gcd2018,gcd2018.factor.class)
str(gcd2018.clean.factor)
gcd2018.clean.factor$Class = NULL
colnames(gcd2018.clean.factor)[21] = 'Class'
str(gcd2018.clean.factor)
summary(gcd2018.clean.factor)

# dplyr gives a good summary of rows andd columns
# it is used for data manipulation
library(dplyr)

# to make a subset of all numeric values. there are 7 columns 
gcd.numeric = select(gcd, Duration, Credit, Rate, Residence, Age, Existing, Liable) 

# standard deviation of gcd.numeric
gcd.numeric %>% summarise_all(funs(sd(.)))

# mean and median of gcd.numeric
summary(gcd.numeric)
# by looking at the standard deviation for Credit we 
# notice its standard deviation is high and it shows that 
# its values is strayed from the mean.
# for other attributes median might be a better option for 
# central tendency


# set random seed:
set.seed(215487)

# create a random sample of rows
train.row = sample(1:nrow(gcd.clean.factor), 0.7 * nrow(gcd.clean.factor))
# to make the train data set
gcd.train = gcd.clean.factor[train.row,]
nrow(gcd.train)
# to make the test data set
gcd.test = gcd.clean.factor[-train.row,]
summary(gcd.test)

# train and test data set from gcd2018 
set.seed(1234567)
train2018.row = sample(1:nrow(gcd2018.clean.factor), 0.7 * nrow(gcd2018.clean.factor))
gcd2018.train = gcd2018.clean.factor[train.row,]
nrow(gcd.train)
gcd2018.test = gcd2018.clean.factor[-train.row,]



library(tree)
library(e1071)
library(ROCR)
library(rpart)
library(adabag)
library(randomForest)
# Classification for decision tree model for gcd data set
set.seed(215487)
library(tree)
gcd.decision_tree = tree(Class ~. , data = gcd.train)
gcd.decision_tree
summary(gcd.decision_tree)
plot(gcd.decision_tree)
text(gcd.decision_tree, pretty = 0)

# Classification for decision tree model for gcd2018 data set
set.seed(215487)
gcd2018.decision_tree = tree(Class ~. , data = gcd2018.train)
gcd2018.decision_tree
summary(gcd2018.decision_tree)
plot(gcd2018.decision_tree)
text(gcd2018.decision_tree, pretty = 0)


# Clasification for Naive Bayes model for gcd data set
set.seed(215487)
library(e1071)
gcd.naive_bayes = naiveBayes(Class ~. , gcd.train)
summary(gcd.naive_bayes)

# Clasification for Naive Bayes model for gcd2018 data set
set.seed(215487)
gcd2018.naive_bayes = naiveBayes(Class ~. , gcd2018.train)
summary(gcd2018.naive_bayes)


# Classification for Bagging model for gcd
library(adabag)
library(rpart)
set.seed(215487)
gcd.bagging = bagging(Class ~ ., data = gcd.train)
summary(gcd.bagging)

# Classification for Bagging model for gcd2018
set.seed(215487)
gcd2018.bagging = bagging(Class ~ ., data = gcd2018.train)
summary(gcd2018.bagging)



# Classification for Boosting Model for gcd
set.seed(24150096)
gcd.boosting = boosting(Class ~., data = gcd.train)
summary(gcd.boosting)


# Classification for Random Forest model for gcd data frame
library(randomForest)
set.seed(215487)
gcd.random_forest = randomForest(Class ~., data = gcd.train)
print(gcd.random_forest)


# Using the test data, classify each of the test 
# cases as ‘Good credit risk’ or ‘Bad credit risk’. 
# Create a confusion matrix and report the accuracy of each model.

# Decision Tree Classification Model for gcd data set
gcd.predict.decision_tree = predict(gcd.decision_tree, gcd.test, type = "class")
cat("\n#Confusion Matrix for Decision Tree Model for gcd Data Set\n")
gcd.decision_tree.confusion_matrix = table(actual = gcd.test$Class, predicted = gcd.predict.decision_tree)
print(gcd.decision_tree.confusion_matrix)

#          predicted
# actual   1    2
#     1   155   14
#     2   49    28
#accuary of mode = (TP + TN)  / (TP +   TN +   FP   + FN)
#                  (155 + 28) /  (155 + 28 +   14   + 49)  =
#                                             183/246 = 0.7439
#                                             = 74.39%


# Decision Tree Classification Model for gcd2018 data set
gcd2018.predict.decision_tree = predict(gcd2018.decision_tree, gcd2018.test, type = "class")
cat("\n#Confusion Matrix for Decision Tree Model for gcd2018 Data Set\n")
gcd2018.decision_tree.confusion_matrix = table(actual = gcd2018.test$Class, predicted = gcd2018.predict.decision_tree)
print(gcd2018.decision_tree.confusion_matrix)
#             predicted
#     actual     1     2
#          1    142    25
#          2    54     24
# accuracy of model is: 0.6587 = 67.75%

# if we use mean as NA fields for age attribute, we get a better 
# accuracy for decision tree classification. the ratio of using mean 
# as NA fields vs remving NA for the age attribute is: 74.39:67.75
# so we choose the better one and it is 74.39%



# Naive Basyes Classification Model for gcd data set
gcd.predict.naive_bayes = predict(gcd.naive_bayes, gcd.test)
cat("\n#Confusion Matrix for Naive Bayes Model for gcd Data Set\n")
gcd.naive_bayes.confusion_matrix = table(actual = gcd.test$Class, predicted = gcd.predict.naive_bayes)
gcd.naive_bayes.confusion_matrix
# accuracy is: (142 + 35) / (142 + 27 + 42 + 35)
# = 177 / 246 = 71.95%

# Naive Basyes Classification Model for gcd2018 data set
gcd2018.predict.naive_bayes = predict(gcd2018.naive_bayes, gcd2018.test)
cat("\n#Confusion Matrix for Naive Bayes Model for gcd2018 Data Set\n")
gcd2018.naive_bayes.confusion_matrix = table(actual = gcd2018.test$Class, predicted = gcd2018.predict.naive_bayes)
gcd2018.naive_bayes.confusion_matrix
# accuracy is: (129 + 47) / (129 + 38 + 31 + 47)
# = 176 / 245 = 71.83%



# Bagging Classification Model for gcd data set
gcd.predict.bagging = predict.bagging(gcd.bagging, gcd.test)
gcd.bagging.confusion_matrix = gcd.predict.bagging$confusion
cat("\n#Confusion Matrix for Bagging Model for gcd Data Set\n")
print(gcd.bagging.confusion_matrix)
# accuracy is: (147 + 46) / (147 + 46 + 22 + 31)
# = 178 / 246 = 72.35% 
# by using mfinal=5 the following accuracy is produced:
# accuracy is: (140 + 30) / (140 + 47 + 29 + 30)
# = 170 / 246 = 69.10%

# Bagging Classification Model for gcd2018 data set
gcd2018.predict.bagging = predict.bagging(gcd2018.bagging, gcd2018.test)
gcd2018.bagging.confusion_matrix = gcd2018.predict.bagging$confusion
cat("\n#Confusion Matrix for Bagging Model for gcd2018 Data Set\n")
print(gcd2018.bagging.confusion_matrix)
# accuracy is: (140 + 40) / (140 + 38 + 27 + 40)
# = 180 / 245 = 73.46%
# by using mfinal=5 the following accuracy is produced:
# accuracy is: (137 + 39) / (137 + 39 + 30 + 39)
# = 176 / 245 = 71.83%


# Boosting Classification Model for gcd data set
gcd.predict.boosting = predict.boosting(gcd.boosting, gcd.test)
gcd.boosting.confusion_matrix = gcd.predict.boosting$confusion
cat("\n#Confusion Matrix for Boosting Model for gcd Data Set\n")
print(gcd.boosting.confusion_matrix)
# accuracy is: (145 + 31) / (145 + 46 + 24 + 31)
# = 176 / 245 = 71.83%
# by using mfinal=5 the following accuracy is produced:
# accuracy is: (146 + 31) / (146 + 46 + 23 + 31)
# = 177 / 245 = 72.24%


# Random Forest Classification Model for gcd data set
gcd.predict.random_forest = predict(gcd.random_forest, gcd.test)
gcd.random_forest.confusion_matrix = table(actual = gcd.test$Class, predicted = gcd.predict.random_forest)
cat("\n#Confusion Matrix for Random Forest Model for gcd Data Set\n")
print(gcd.random_forest.confusion_matrix)
# accuracy is: (156 + 27) / (156 + 13 + 50 + 27)
# = 183 / 245 = 74.39%


# Using the test data, calculate the confidence of predicting a 
# ‘Good credit risk’ for each case and construct an ROC curve for 
# each classifier. 
# making ROC curve for each of the classifiers
# then calculate thr AUC for each classifier

library(tree)
set.seed(215487)
gcd.decision_tree = tree(Class ~. , data = gcd.train)

library(ROCR)

# Decision Tree Classification Model for gcd data set
# calculatinf the confidence value for Decision Tree classification for gcd data frame
gcd.confidence.decision_tree = predict(gcd.decision_tree, gcd.test, type = "vector")
# calculating the ROC curve for Decision Tree classification for gcd data frame
gcd.predict.con.decision_tree = prediction(gcd.confidence.decision_tree[,1], gcd.test$Class, label.ordering = c('2', '1'))
# Constructing the ROC curve for Decision Tree classifier (x: fpr, y: tpr)
gcd.performance.decision_tree = performance(gcd.predict.con.decision_tree,"tpr","fpr")
plot(gcd.performance.decision_tree, col = 'green', main = "Classifier ROC Curves")
abline(0, 1)
# calculating the AUC for Decision Tree Classifier for gcd data set
gcd.auc.decision_tree = performance(gcd.predict.con.decision_tree, "auc")
gcd.auc.decision_tree.numeric = as.numeric(gcd.auc.decision_tree@y.values)
gcd.auc.decision_tree.numeric
# 67.17%


# Naive Bayes Classification Model for gcd data set
# calculating the confidence value for Naive Bayes classification for gcd data frame
gcd.confidence.naive_bayes = predict(gcd.naive_bayes, gcd.test, type = "raw")
# calculating the ROC curve for  Naive Bayes for gcd data frame
gcd.predict.con.naive_bayes = prediction(gcd.confidence.naive_bayes[,1], gcd.test$Class, label.ordering = c('2', '1'))
# Constructing the ROC curve for Naive Bayes classifier (x: fpr, y: tpr)
gcd.performance.naive_bayes = performance(gcd.predict.con.naive_bayes,"tpr","fpr")
plot(gcd.performance.naive_bayes,add = TRUE, col = 'red', main = "Classifier ROC Curves")
abline(0, 1)
# calculating the AUC for Decision Tree Classifier for gcd data set
gcd.auc.naive_bayes = performance(gcd.predict.con.naive_bayes, "auc")
gcd.auc.naive_bayes.numeric = as.numeric(gcd.auc.naive_bayes@y.values)
gcd.auc.naive_bayes.numeric
# 73.88%


# Bagging Classification Model for gcd data set
# calculating the ROC curve for  Bagging for gcd data frame
gcd.predic.confusion.bagging = gcd.predict.bagging$confusion
gcd.predic.confusion.bagging
gcd.predict.bagging = prediction(gcd.predict.bagging$prob[,1], gcd.test$Class, label.ordering = c('2', '1'))
# Constructing the ROC curve for Bagging classifier (x: fpr, y: tpr)
gcd.performance.bagging = performance(gcd.predict.bagging,"tpr","fpr")
plot(gcd.performance.bagging, add = TRUE, col = 'orange', main = "Classifier ROC Curves")
abline(0, 1)
# calculating the AUC for bagging Classifier for gcd data set
gcd.auc.bagging = performance(gcd.predict.bagging, "auc")
gcd.auc.bagging.numeric = as.numeric(gcd.auc.bagging@y.values)
gcd.auc.bagging.numeric
# 73.84%


# boosting Classification Model for gcd data set
# calculating the ROC curve for  boosting for gcd data frame
gcd.predict.boosting = prediction(gcd.predict.boosting$prob[,1], gcd.test$Class, label.ordering = c('2', '1'))
# Constructing the ROC curve for boosting classifier (x: fpr, y: tpr)
gcd.performance.boosting = performance(gcd.predict.boosting,"tpr","fpr")
plot(gcd.performance.boosting, add = TRUE, col = 'purple', main = "Classifier ROC Curves")
abline(0, 1)
# calculating the AUC for boosting Classifier for gcd data set
gcd.auc.boosting = performance(gcd.predict.boosting, "auc")
gcd.auc.boosting.numeric = as.numeric(gcd.auc.boosting@y.values)
gcd.auc.boosting.numeric
# 76.56%


# random forest Classification Model for gcd data set
gcd.confidence.random_forest = predict(gcd.random_forest, gcd.test, type = "prob")
# calculating the ROC curve for  random forest for gcd data frame
gcd.predict.random_forest = prediction(gcd.confidence.random_forest[,1], gcd.test$Class, label.ordering = c('2', '1'))
# Constructing the ROC curve for random forest classifier (x: fpr, y: tpr)
gcd.performance.random_forest = performance(gcd.predict.random_forest,"tpr","fpr")
plot(gcd.performance.random_forest, add = TRUE, col = 'yellow', main = "Classifier ROC Curves")
abline(0, 1)
# calculating the AUC for random forest Classifier for gcd data set
gcd.auc.random_forest = performance(gcd.predict.random_forest, "auc")
gcd.auc.random_forest.numeric = as.numeric(gcd.auc.random_forest@y.values)
gcd.auc.random_forest.numeric
# 76.58


# Examining each of the models, determine the most 
# important variables in predicting whether or not an 
# applicant is a good or bad credit risk.
cat("\n # important attributes for Decision Tree classification \n")
summary(gcd.decision_tree)

cat("\n #important attributes for Naive Bayes classification \n")
summary(gcd.naive_bayes)

cat("\n #important attributes for Bagging classification \n")
print(gcd.bagging$importance)

cat("\n #important attributes for Boosting classification \n")
print(gcd.boosting$importance)

cat("\n #important attributes for Random Forest classification \n")
print(gcd.random_forest$importance)


# By experimenting with parameter settings 
# for at least one of the classifiers, create 
# the best classifier you can – that is, one with an 
# accuracy greater than the models you originally created 
# in Part A. Demonstrate this improved accuracy using ROC, AUC, 
# or other accuracy measures. Report the parameter settings and 
# assumptions made in designing this classifier. 
set.seed(215487)
mtry = tuneRF(gcd.train[,-21], gcd.train$Class, ntreeTry=301, stepFactor=1.5, improve=0.05)
best.m = mtry[mtry[, 2] == min(mtry[, 2]), 1]
best.m

library(randomForest)
gcd.random_forest.improved = randomForest(Class ~., data = gcd.train, mtry = best.m, ntree = 441, importance=TRUE)
print(gcd.random_forest)


gcd.confidence.random_forest.improved = predict(gcd.random_forest.improved, gcd.test, type = "prob")
# calculating the ROC curve for  random forest for gcd data frame
gcd.predict.random_forest.improved = prediction(gcd.confidence.random_forest.improved[,1], gcd.test$Class, label.ordering = c('2', '1'))
# Constructing the ROC curve for random forest classifier (x: fpr, y: tpr)
gcd.performance.random_forest.improved = performance(gcd.predict.random_forest.improved,"tpr","fpr")
plot(gcd.performance.random_forest, add = TRUE, col = 'pink', main = "Classifier ROC Curves")
abline(0, 1)

# calculating the AUC for random forest Classifier for gcd data set
gcd.auc.random_forest.improved = performance(gcd.predict.random_forest.improved, "auc")
gcd.auc.random_forest.improved.numeric = as.numeric(gcd.auc.random_forest.improved@y.values)
gcd.auc.random_forest.improved.numeric


# Using the insights from your analysis 
# so far, implement an Artificial Neural Network 
# classifier and report its performance.
# attributes used and your data pre-processing required. 
library(neuralnet)
gcd.neural = gcd
set.seed(215487)

gcd.neural.class.clean = gcd.neural[!(is.na(gcd.neural$Class)),]
gcd.neural.clean.com = gcd.neural.class.clean[complete.cases(gcd.neural.class.clean),]

# make training and test sets
ind = sample(2, nrow(gcd.neural.clean.com), replace = TRUE, prob=c(0.7, 0.3))
# ind
gcd.neural.train = gcd.neural.clean.com[ind == 1,]
gcd.neural.test = gcd.neural.clean.com[!ind == 1,]


gcd.neural_network = neuralnet(Class ~ Duration + Credit + Rate + Residence + Age + Existing + Liable, gcd.neural.train, hidden=3)
gcd.neural.predict = compute(gcd.neural_network, gcd.neural.test[c(2, 5, 8, 11, 13, 16, 18)])

# now round these down to integers 
gcd.neural.predict.frame = as.data.frame(round(gcd.neural.predict$net.result,0))

# plot confusion matrix 
table(observed = gcd.neural.test$Class, predicted = gcd.neural.predict.frame$V1)






