
## Set Working Directory

setwd("D:/BABI/Predictive Modelling/Churn Analysis Project/")
getwd()

library(psych)
library(DataExplorer)
library(corrplot)
library(ppcor)
library(rpart)
library(rpart.plot)
library(ROCR)
library(rattle)
library(car)
library(olsrr)
library(MASS)
library(class)
library(caret)
library(lattice)
library(ggplot2)

#Import dataset from File menu - File name -> Cellphone_Modified.xlsx
#The data is loaded onto mydata object
library(readxl)
mydata <- read_excel("Cellphone _Modified.xlsx")
View(mydata)



#Importing the excel file  the data and descriptive statistics

head(mydata)
dim(mydata)
str(mydata)
names(mydata)
describe(mydata)
summary(mydata)

## Check for missing value (NA)
anyNA(mydata)
plot_missing(mydata)


attach(mydata)

## Plotting histogram to understand the overview of the data


plot_histogram(mydata)


## Plotting boxplot for all independant variables

boxplot(AccountWeeks,DataUsage,CustServCalls,DayMins,
        main = "Multiple boxplots for comparision",
        names = c("AccountWeeks","Data Usage","CustCare Call", "Day Minutes"),
        border = "brown")

boxplot(DayCalls,MonthlyCharge,OverageFee,RoamMins,
        main = "Multiple boxplots for comparision",
        names = c("Day Calls","Monthly Charge",
                  "Overag eFee ","Roaming Minutes"),
        border = "brown")

## Verifying correlation - since the modified excel sheet has multiple factor column
## we cannot perform correleation analysis since correlation in R works mainly on continous 
##variable. So we load the original dataset for corrleation and multicolienarity check .
mydata1 <- read_excel("Cellphone.xlsx")
View(mydata1)
plot_correlation(mydata1)

library(corrplot)
datamatrix<-cor(mydata1)
corrplot(datamatrix, method ="number")

#Pairwise correlation
# For examining the patterns of multicollinearity, it is required to conduct t-test for correlation coefficient. 
# ppcor package helps to compute the partial correlation coefficients along with the t-statistics and corresponding p values for the independent variables.

pcor(mydata1, method = "pearson")

#Initial Regression Model using the data as it is

model0 = lm(Churn~., mydata1)
summary(model0)
vif(model0)

# Performing Step regression to remove insignificant variables 
ols_step_both_p(model0, pent = 0.05, prem = 0.3, details = TRUE)

#new model with reduced variables
model1 = lm(Churn~ContractRenewal+CustServCalls+
              DayMins+OverageFee+DataPlan+RoamMins, mydata1)
summary(model1)
vif(model1) # VIF is around 1 for all variables 

## Multicolinearity is removed 

cleandata <- subset(mydata1, select = -c(2,5,8,9))
str(cleandata)

## Converting continous to categorical variables

cleandata$Churn = as.factor(cleandata$Churn)
cleandata$ContractRenewal = as.factor(cleandata$ContractRenewal)

anyNA(cleandata)
dim(cleandata)
plot_correlation(cleandata)
names(cleandata)

attach(cleandata)

## We have completed the EDA and the dataset is redy for further modelling 

## Spliting the dataset into train and test for development and out of sample testing respectively
set.seed(100)
TRAIN_INDEX <- sample(1:nrow(cleandata),0.75*nrow(cleandata))
TrainData <- cleandata[TRAIN_INDEX,]
TestData <- cleandata[-TRAIN_INDEX,]

summary(TestData)
summary(TrainData)

dim(TrainData)
dim(TestData)

head(TrainData)
head(TestData)

#**********************************************************************
#*********************************************************************

# Perform Logistic Regression

LogitModel1 <- glm(Churn ~ ., data = TrainData,family = binomial(link = 'logit'))
print(LogitModel1)
summary(LogitModel1)


## CIs using profiled log-likelihood
confint(LogitModel1)

exp(coef(LogitModel1))

## odds ratios and 95% CI
exp(cbind(OR = coef(LogitModel1), confint(LogitModel1)))

##Fitting the same model on test data 
names(TrainData)


TrainData
LogitModel2 <- glm(Churn ~ ., data = TestData,family = binomial(link = 'logit'))
summary(LogitModel2)


################################################################
###############################################################
## a function for error rate
get_Error_Rate=function(trues, predicted_prb, cutoff){
        preds=ifelse(predicted_prb<cutoff,0,1)
        tab=table(preds, trues)
        round((tab[1,2]+tab[2,1])/sum(tab), 4)
}

get_Error_Rate(TrainData$Churn,LogitModel1$fitted.values, 0.5)

get_Error_Rate(TestData$Churn,LogitModel2$fitted.values, 0.5)

#Confusion matrix / accuracy

##Define Accuracy function

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

pred_train = predict(LogitModel1, newdata=TrainData[,-1],type="response") 
tab1=table(TrainData$Churn, pred_train>0.5) 
accuracy(tab1)

tab1
# Confussion Matrix on Test Data #
pred_test = predict(LogitModel1, newdata=TestData[,-1], type="response") 
tab2=table(TestData$Churn, pred_test>0.5)
tab2
accuracy(tab2)

## Area Under the ROC curve (AUC - ROC) ##
#Validation on train data 
library(ROCR)
predictROC1 = predict(LogitModel1, newdata = TrainData)
pred1 = prediction(predictROC1, TrainData$Churn)
perf1 = performance(pred1, "tpr", "fpr") 
plot(perf1,colorize =T) 
as.numeric(performance(pred1, "auc")@y.values)

#Validation on test data
predictROC2 = predict(LogitModel1, newdata = TestData) 
pred2 = prediction(predictROC2, TestData$Churn) 
perf2 = performance(pred2, "tpr", "fpr") 
plot(perf2,colorize =T) 
as.numeric(performance(pred2, "auc")@y.values)
# KS Test Validation #
library(ineq)
#KS on train
KSLRTrain = max(attr(perf1, 'y.values')[[1]]-attr(perf1, 'x.values')[[1]])
KSLRTrain
#KS on test
KSLRTest =  max(attr(perf2, 'y.values')[[1]]-attr(perf2, 'x.values')[[1]]) 
KSLRTest
# Gini Test Validation #
#Gini for Train
giniLRTrain = ineq(pred_train, type="Gini")
giniLRTrain
#Gini for Test#
giniLRTest = ineq(pred_test, type="Gini")
giniLRTest

###################################################################
#######################################################################




## KNN Model - data normalization, remove the Churn variable

#Normalization
normalize <- function(x) 
{ return ((x - min(x)) / (max(x) - min(x)))}


#KNNData <- subset(cleandata, select = -c(1))
KNNData=mydata1


# To normalize , need to convert ContractRenewal to numeric field
KNNData$ContractRenewal = as.numeric(KNNData$ContractRenewal)
KNNData$Churn = as.numeric(KNNData$Churn)
str(KNNData)
KNNData <- as.data.frame(lapply(KNNData, normalize))
head(KNNData)

## Spliting the dataset into train and test for development and out of sample testing respectively


set.seed(123)
dat.d <- sample(1:nrow(KNNData),size=nrow(KNNData)*0.7,replace = FALSE) #random selection of 70% data.


train.churn <- KNNData[dat.d,] # 70% training data
test.churn <- KNNData[-dat.d,] # remaining 30% test data

str(train.churn)
str(test.churn)

#Creating seperate dataframe for 'Churn' feature which is our target.
train.churn_labels <- KNNData[dat.d,1]
test.churn_labels <-KNNData[-dat.d,1]

NROW(train.churn_labels)
NROW(test.churn_labels)

target<-as.factor(KNNData[dat.d,1])
testtarget<-as.factor(KNNData[-dat.d,1])

testtarget
## Build KNN model

KNN.49<- knn(train=train.churn, test=test.churn, cl=train.churn$Churn, k=5)
KNN.50<- knn(train=train.churn, test=test.churn, cl=train.churn$Churn, k=50)


summary(KNN.49)
summary(KNN.50)

tb1 <- table(KNN.49,testtarget)
tb2<-table(KNN.50,testtarget)

tb1
tb2

##check the accuracy
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tb1)
accuracy(tb2)

KNN.49_A<- knn(train=train.churn[,c(3,4,6,7)], test=test.churn[,c(3,4,6,7)], cl=train.churn$Churn, k=49)
tb1_A <- table(KNN.49_A,testtarget)
accuracy(tb1_A)

#**********************************************************************
#*********************************************************************

## Applying Naive Bayes

library(e1071)
## Spliting the dataset into train and test for development and out of sample testing respectively
set.seed(100)
cleandata=mydata1
cleandata$Churn=as.factor(cleandata$Churn)
TRAIN_INDEX <- sample(1:nrow(cleandata),0.75*nrow(cleandata))
TrainData <- cleandata[TRAIN_INDEX,]
TestData <- cleandata[-TRAIN_INDEX,]
head(TrainData)
NBModel=naiveBayes(as.factor(TrainData$Churn) ~., data = TrainData,method="class")
str(NBModel)
NBModel

#Confusion Matrix and prediction on Train data 
predict1 = predict(NBModel, newdata=TrainData, type = "class")
tab3=table(TrainData$Churn, predict1)
tab3
accuracy(tab3)

#Confusion Matrix and prediction on Train data 
predict2 = predict(NBModel, newdata=TestData, type = "class")
tab4=table(TestData$Churn, predict2)
tab4
accuracy(tab4)


