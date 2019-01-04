###########################
## Script for Homework 2 ##
###########################

rm(list=ls()); gc()
setwd('C:\\Users\\bdardin\\Documents\\Datasets')

dat = read.csv('toyota_clean2.csv', head=T, stringsAsFactors=F) 

###########################
##  Data Preprocessing   ##
###########################

#Get the 75th percentile of the Price column in order to dichotomize it
quant = quantile(dat$Price, probs = c(0.75))

#Create a new column that is 1 if the row >= 75th percentile and 0 if not
high_price = c()
for(i in 1:dim(dat)[1]) {
  val = 0
  if(dat$Price[i] >= quant) {
    val = 1
  }
  high_price <- append(high_price,val)
}

#Create new dataframe with this new column and without the original price
dat2 <- cbind(high_price,dat)
dat2 <- dat2[,-which(colnames(dat2) == 'Price')]

#Split the data into 60% training and 40% validation
#The number of rows hasn't changed so the same indices can be used for both
set.seed(12345)
id.train = sample(1:nrow(dat), nrow(dat)*.6)
id.test = setdiff(1:nrow(dat), id.train)

# assign train/test x/y splits into variables to improve readability
x_train = dat[id.train,2:dim(dat)[2]]
x_test = dat[id.test,2:dim(dat)[2]]

price_train = dat[id.train,1]
price_test = dat[id.test,1]
high_price_train = dat2[id.train,1]
high_price_test = dat2[id.test,1]

###########################
##    Target = Price     ##
###########################

## linear regression ##

#Create initial regression models with no variables and all variables
lin_reg.null = lm(Price ~ 1, dat = dat[id.train, ]) #none
lin_reg.full = lm(Price ~ ., dat = dat[id.train, ]) #all

# forward selection
lin_reg_fwd = step(lin_reg.null, scope=list(lower=lin_reg.null, upper=lin_reg.full), direction='forward')

# RMSE
require(hydroGOF)

lin_reg_yhat = predict(lin_reg_fwd, newdata = dat[id.test, ])
lin_reg_rmse = rmse(price_test, lin_reg_yhat)

# for the simplicity measure, we look at the number of variables used as coefficients
# as well as look at the percentage of the variables used
lin_reg_var = length(lin_reg_fwd$coefficients) - 1
total_variables = (dim(dat)[2] - 1)
lin_reg_pct = lin_reg_var / total_variables

# print the metrics
cat("Linear Regression\nRMSE:",lin_reg_rmse,"\nVariables:",lin_reg_var,"\nPercent Used:",lin_reg_pct)

## regression-based kNN ##

require(class)

# I modified the original function to add a new parameter which tells the function
# whether to use regression or classification based kNN for the calculations
# it assumes classification as default true, change to false for regression
# I also changed the return value to include the best model so it doesn't have to be rerun
knn.bestK = function(train, test, y.train, y.test, k.max = 20, classif = TRUE) {
  k.grid = seq(1, k.max, 2) 
  error = rep(NA, length(k.grid))
  models = list()
  if(classif) {
    for (ii in k.grid) {
      # the classification steps are left the same
      y.hat = knn(train, test, y.train, k = ii, prob=F) 
      error[(ii+1)/2] = sum(y.hat != y.test)
      models[[(ii+1)/2]] = y.hat
    }
  } else { # if regression
    for (ii in k.grid) {
      # run regression-based kNN
      reg_model = FNN::knn.reg(train, test, y.train, k = ii)
      # use RMSE as the error metric
      error[(ii+1)/2] = rmse(y.test, reg_model$pred)
      models[[(ii+1)/2]] = reg_model
    }
  }
  out = list(k.optimal = k.grid[which.min(error)], error.min = min(error), model.min = models[[which.min(error)]])
  return(out)
}

best_k_reg = knn.bestK(x_train, x_test, price_train, price_test, classif = FALSE)

# print the metrics, here we use the chosen k as the simplicity metric
cat("Regression kNN\nRMSE:",best_k_reg$error.min,"\nBest K:",best_k_reg$k.optimal)

## regression tree ##

tree_reg = rpart(Price ~ ., method="anova", data=dat[id.train, ])
tree_reg_yhat = predict(tree_reg, x_test)
tree_rmse = rmse(price_test, tree_reg_yhat)

# for the simplicity metric we look at how many nodes there are, minus the leaves
tree_reg_nodes = sum(tree_reg$frame$var != "<leaf>")
# we also look at how many variables are ultimately used in the nodes
tree_reg_var = length(levels(factor(tree_reg$frame$var))) - 1

# print the metrics
cat("Regression Tree\nRMSE: ",tree_rmse,"\nNodes:",tree_reg_nodes,"\nVariables:",tree_reg_var)

###########################
##  Target = High_Price  ##
###########################

# create a simple function to calculate the accuracy
accuracy = function(predicted, expected) {
  return( sum(predicted == expected) / length(expected) )
}

## logistic regression ##

# like with the linear regression model, we initially train 2 models, one with no variables and one with all
log.min = glm(high_price ~ 1, data = dat2[id.train, ], family = 'binomial') #none
log.max = glm(high_price ~ ., data = dat2[id.train, ], family = 'binomial') #all
log.formula = formula(log.max)

# forward selection
log_reg = step(log.min, direction='forward', scope=log.formula)

# generate the probabilities, then use cutoff of .5 to create class predictions
log_yhat_prob = predict(log_reg, x_test, type='response')
log_yhat_class = as.integer(log_yhat_prob > .5)
log_acc = accuracy(log_yhat_class, high_price_test)

# for the simplicity measure, we look at the number of variables used as coefficients
# as well as look at the percentage of the variables used
log_var = length(log_reg$coefficients) - 1
log_pct = log_var / total_variables

# print the metrics
cat("Logistic Regression\nAccuracy:",log_acc,"\nVariables:",log_var,"\nPercent Used:",log_pct)

## classification-based kNN ##

best_k_class = knn.bestK(x_train, x_test, high_price_train, high_price_test)
knn_acc = accuracy(best_k_class$model.min, high_price_test)

# print the metrics, here again we use the chosen k as the simplicity metric
cat("Classification kNN\nAccuracy:",knn_acc,"\nBest K:",best_k_class$k.optimal)

## classification tree ##

tree_class = rpart(high_price ~ ., method="class", data=dat2[id.train, ])
tree_class_pred = predict(tree_class, x_test, type = "class")
tree_acc = accuracy(tree_class_pred, high_price_test)

# for the simplicity metric we again look at the number of nodes and variables used
tree_class_nodes = sum(tree_class$frame$var != "<leaf>")
tree_class_var = length(levels(factor(tree_class$frame$var))) - 1

# print the metrics
cat("Classification Tree\nAccuracy:",tree_acc,"\nNodes:",tree_class_nodes,"\nVariables:",tree_class_var)

# compute the MAPE for the linear regression model and error for the classification tree
mape = mean(abs(price_test-lin_reg_yhat)/price_test) * 100
error = (1 - tree_acc) * 100
cat("MAPE:",mape,"\nError Rate:",error)