#######################
## Script for Homework 1 ##
#######################

rm(list=ls()); gc()
setwd('C:\\Users\\bdardin\\Documents\\Datasets')

dat = read.csv('ToyotaCorolla_new.csv', head=T, stringsAsFactors=F) 

#######################
## data cleaning ##
#######################

#The dataset was pre-cleaned by the professor

#######################
## split & export data ##
#######################

#Convert the categorical Color variable into dummies for exporting to XLMiner
colors = factor(dat5$Color)
dummies = model.matrix(~colors)
dat6 = cbind(dat5,dummies[,2:7])
dat6 = dat6[,-which(colnames(dat6) == 'Color')] #remove the original color variable

#Split the data into 60% training and 40% validation
set.seed(1)
id.train = sample(1:nrow(dat6), nrow(dat6)*.6)
id.test = setdiff(1:nrow(dat6), id.train)

#Create new column that is 1 if the row is in the training set and 0 if not
train_col = c()
for(i in 1:dim(dat6)[1]) {
  val = 0
  if(i %in% id.train) {
    val = 1
  }
  train_col <- append(train_col,val)
}

#Create new dataframe with this new column
dat7 <- cbind(train_col,dat6)

# Write CSV in R
write.csv(dat7, file = "Train_Test_Data.csv")

#######################
## linearity check ##
#######################

library(MASS)

#Look at each non-binary variable and its relationship to price
plot(dat6$Age_08_04, dat6$Price)
plot(dat6$Weight, dat6$Price)
plot(dat6$Quarterly_Tax, dat6$Price)
plot(dat6$HP, dat6$Price)
plot(dat6$CC, dat6$Price)

#######################
## model creation ##
#######################

#Create initial regression models with no variables and all variables
obj.null = lm(Price ~ 1, dat = dat6[id.train, ]) #none
obj.full = lm(Price ~ ., dat = dat6[id.train, ]) #all

## forward selection ##
obj1 = step(obj.null, scope=list(lower=obj.null, upper=obj.full), direction='forward')
summary(obj1)

# normality (of the residual)
hist(obj1$resid)

# Homoscedasticity
plot(obj1$resid, obj1$fitted)

## backward selection ##
obj2 = step(obj.full, scope=list(lower=obj.null, upper=obj.full), direction='backward')
summary(obj2)

# normality (of the residual)
hist(obj2$resid)

# Homoscedasticity
plot(obj2$resid, obj2$fitted)

## stepwise selection ##
obj3 = step(obj.null, scope=list(lower=obj.null, upper=obj.full), direction='both')
summary(obj3)

# normality (of the residual)
hist(obj3$resid)

# Homoscedasticity
plot(obj3$resid, obj3$fitted)

#######################
## model evaluation ##
#######################

require(hydroGOF)

# forward
yhat1 = predict(obj1, newdata = dat6[id.test, ])
rmse(dat6[id.test, 'Price'], yhat1)

# backward
yhat2 = predict(obj2, newdata = dat6[id.test, ])
rmse(dat6[id.test, 'Price'], yhat2)

# stepwise
yhat3 = predict(obj3, newdata = dat6[id.test, ])
rmse(dat6[id.test, 'Price'], yhat3)