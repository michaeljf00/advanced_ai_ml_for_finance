rm(list=ls())
# Load libraries
library("quantmod")
library("PerformanceAnalytics")
library("glmnet")
library("caret")

# Set seed for reproducibility
set.seed(19740284)

# Get S&P 500 data
t1 <- "1980-01-01"
getSymbols("SPY", from = t1, to = Sys.Date(), adjust = TRUE)

# Assign outcome variable as next period returns
volume <- SPY[,5]
names(volume) <- c("SPY_vol")
adjusted <- SPY[,6]
names(adjusted) <- c("SPY")
next_period_returns <- na.omit(Return.calculate(adjusted))

# Retrieve and engineer features
indices <- c("UNRATE", "DFF", "CORESTICKM159SFRBATL", "INDPRO", "BAMLH0A0HYM2", "FEDFUNDS", "MEDCPIM158SFRBCLE", "DEXUSEU")
ind.list <- lapply(indices, function(tic) get(getSymbols(tic, from = "2000-01-01", src = "FRED")))
IND <- na.omit(Reduce(merge, ind.list))

# 1 week rolling average
days <- 7
return_roll <- rollapply(next_period_returns, days, mean)
names(return_roll) <- paste("SPY", "_roll_", days, sep="")

# Lag
SPY_lag <- lag(next_period_returns$SPY, 1)
names(SPY_lag) <- "SPY_lag"

# Merge features and target variable
df_np_returns <- na.omit(merge(IND, volume, return_roll, SPY_lag, next_period_returns))

# Data Pre-processing
sum(is.na(df_np_returns))

# Standardize features
std_vars <- matrix(rep(apply(df_np_returns, 2, sd), nrow(df_np_returns)), ncol = ncol(df_np_returns), byrow = TRUE)
standardized <- df_np_returns / std_vars

my.dates <- as.Date(index(standardized))
head(my.dates)

d1 <- min(my.dates)
d2 <- as.Date("2008-02-01")
d3 <- as.Date("2016-02-01")

idx1 <- which(my.dates >= d1 & my.dates <= d2)
idx2 <- which(my.dates > d2 & my.dates <= d3)
idx3 <- which(my.dates > d3)

head(my.dates[idx1],3)
tail(my.dates[idx1],3)

head(my.dates[idx2],3)
tail(my.dates[idx2],3)

head(my.dates[idx3],3)
tail(my.dates[idx3],3)

train_data <- standardized[idx1,] # train
validation_data <- standardized[idx2,] # validation
test_data <- standardized[idx3,] # test

# Define features and target variables
features <- colnames(standardized)[1:11]
target <- "SPY"

train <- train_data[, features]
validate <- validation_data[, features]
test <- test_data[, features]

train_target <- train_data[, target]
validate_target <- validation_data[, target]
test_target <- test_data[, target]

# Model implementation
# grid1 <- 10^seq(10, -2, length = 100)
# grid2 <- 10^seq(15, 3, length = 100)

# Function to calculate R-squared
R2 <- function(pred,act){
  rss <- sum((pred - act) ^ 2)  ## residual sum of squares
  tss <- sum((act - mean(act)) ^ 2)  ## total sum of squares
  rsq <- 1 - rss/tss
}

R2_gu <- function(pred,act){
  rss <- sum((pred - act) ^ 2)  ## residual sum of squares
  tss <- sum((act - 0) ^ 2)  ## Note the TSS is NOT normalized by mean
  rsq <- 1 - rss/tss
}


# Function to calculate Mean Squared Error (MSE)
mse <- function(target, predicted) {
  mean((as.vector(target) - as.vector(predicted))^2)
}

####################################################################################################

# Ridge Regression
ridge_model1 <- glmnet(train, train_target,
                              alpha = 0,
                              nlambda=100,
                              standardize = F,
                              thresh=1e-16)

ridge_model1$lambda
coef(ridge_model1)

# Predictions for each lambda 
valid.hat <- predict.glmnet(ridge_model1,
                            newx=as.matrix(validate),
                            s = ridge_model1$lambda)

# 100 models (lambdas)
dim(valid.hat)

# Compute the R2 of each model to select the correct lambdas:
validate1_R2 <- apply(valid.hat,2,function(x){R2(x,validate_target)}) # For each of the 100 values, the rsquared is calculated
validate1_R2_gu <- apply(valid.hat,2,function(x){R2_gu(x,validate_target)})

plot(validate1_R2)
plot(validate1_R2_gu)

ridge1_lambda <- which.max(validate1_R2)
ridge1_lambda

ridge1_opt_lambda <- ridge_model1$lambda[ridge1_lambda]
ridge1_opt_lambda

# Make final predictions:

# Combine training and validation samples:
train1_final <- rbind(train,validate)
target1_final <- rbind(train_target,validate_target)
dim(train1_final)
dim(target1_final)

ridge1_final <- glmnet(train1_final,target1_final,
                       alpha=0,
                       lambda=ridge1_opt_lambda,
                       standardize = F,
                       thresh=1e-16)

ridge1_final
summary(ridge1_final)
str(ridge1_final)

ridge1_pred <- predict.glmnet(ridge1_final, newx=as.matrix(test), s=ridge1_opt_lambda)

dim(ridge1_pred)

ridge1_final_R2 <- apply(ridge1_pred,2,function(x){R2(x,test_target)})
ridge1_final_R2

ridge1_mse <- mse(test_target, ridge1_final_R2)
ridge1_mse

# Overfitting lambda
ridge1_lambda_overfit <- which.min(ridge_model1$lambda) 

ridge1_bad_lambda <- ridge_model1$lambda[ridge1_lambda_overfit] # value of lambda is the maximum rqsquared value
ridge1_bad_lambda

ridge1_bad <- glmnet(train1_final,target1_final,
                     alpha=0,
                     lambda=ridge1_bad_lambda,
                     standardize = F,
                     thresh=1e-16)

ridge1_bad
summary(ridge1_bad)
str(ridge1_bad)


# Finally, make our final predicitons in the test sample:
ridge1_final_bad <- predict.glmnet(ridge1_bad,
                                   newx=as.matrix(test),
                                   s = ridge1_bad_lambda)

names(ridge1_final_bad) <- c("OverFit")

# Calculate the R2 of each model to select the correct lambdas:
ridge1_final_badR2<- apply(ridge1_final_bad,2,function(x){R2(x,test_target)})
ridge1_final_R2
ridge1_final_badR2

ridge1_mse2 <- mse(test_target, ridge1_final_bad)
ridge1_mse2

# Ridge Coefficients
ridge_coefficients1 <- as.matrix(coef(ridge1_final, s = "lambda.min"))
ridge_coefficients2 <- as.matrix(coef(ridge1_bad, s = "lambda.min"))

selected_features1 <- rownames(ridge_coefficients1)[ridge_coefficients1 != 0]
selected_features2 <- rownames(ridge_coefficients2)[ridge_coefficients2 != 0]

cat("Selected Features:", selected_features1, "\n")
cat("Selected Features:", selected_features2, "\n")

####################################################################################################

# Lasso Regression
lasso_model1 <- glmnet(train,train_target,
                             alpha=1,
                             nlambda=100,
                             standardize = F,
                             thresh=1e-16)

lasso_model1$lambda
coef(lasso_model1)

# Predictions for each lambda
valid.hat <- predict.glmnet(lasso_model1,
                            newx=as.matrix(validate),
                            s = lasso_model1$lambda)

# 100 models (lambdas)
dim(valid.hat)

# Compute the R2 of each model to select the correct lambdas:
validate1_R2 <- apply(valid.hat,2,function(x){R2(x,validate_target)}) # For each of the 100 values, the rsquared is calculated
validate1_R2_gu <- apply(valid.hat,2,function(x){R2_gu(x,validate_target)})

plot(validate1_R2)
plot(validate1_R2_gu)

lasso1_lambda <- which.max(validate1_R2)
lasso1_lambda

lasso1_opt_lambda <- lasso_model1$lambda[lasso1_lambda]
lasso1_opt_lambda

# Make final predictions:

# Combine training and validation samples:
train1_final <- rbind(train,validate)
target1_final <- rbind(train_target,validate_target)
dim(train1_final)
dim(target1_final)

lasso1_final <- glmnet(train1_final,target1_final,
                      alpha=1,
                      lambda=lasso1_opt_lambda,
                      standardize = F,
                      thresh=1e-16)

lasso1_final
summary(lasso1_final)
str(lasso1_final)

lasso1_pred <- predict.glmnet(lasso1_final, newx=as.matrix(test), s=lasso1_opt_lambda)

dim(lasso1_pred)

lasso1_final_R2 <- apply(lasso1_pred,2,function(x){R2(x,test_target)})
lasso1_final_R2

lasso1_mse <- mse(test_target, lasso1_final_R2)
lasso1_mse

# Overfitting lambda
lasso1_lambda_overfit <- which.min(lasso_model1$lambda) 

lasso1_bad_lambda <- lasso_model1$lambda[lasso1_lambda_overfit] # value of lambda is the maximum rqsquared value
lasso1_bad_lambda

lasso1_bad <- glmnet(train1_final,target1_final,
                    alpha=1,
                    lambda=lasso1_bad_lambda,
                    standardize = F,
                    thresh=1e-16)

lasso1_bad
summary(lasso1_bad)
str(lasso1_bad)


# Finally, make our final predicitons in the test sample:
lasso1_final_bad <- predict.glmnet(lasso1_bad,
                                  newx=as.matrix(test),
                                  s = lasso1_bad_lambda)

names(lasso1_final_bad) <- c("OverFit")

# Calculate the R2 of each model to select the correct lambdas:
lasso1_final_badR2<- apply(lasso1_final_bad,2,function(x){R2(x,test_target)})
lasso1_final_R2
lasso1_final_badR2

lasso1_mse2 <- mse(test_target, lasso1_final_bad)
lasso1_mse2

# Lasso Coefficients
lasso_coefficients1 <- as.matrix(coef(lasso1_final, s = "lambda.min"))
lasso_coefficients2 <- as.matrix(coef(lasso1_bad, s = "lambda.min"))

selected_features1 <- rownames(lasso_coefficients1)[lasso_coefficients1 != 0]
selected_features2 <- rownames(lasso_coefficients2)[lasso_coefficients2 != 0]

cat("Selected Features:", selected_features1, "\n")
cat("Selected Features:", selected_features2, "\n")

####################################################################################################

# Elastic Net Regression
elastic_net_model1 <- glmnet(train,train_target,
                      alpha=0.6,
                      nlambda=100,
                      standardize = F,
                      thresh=1e-16)

elastic_net_model1$lambda 
coef(elastic_net_model1)


# Predictions for each lambda
valid.hat <- predict.glmnet(elastic_net_model1,
                            newx=as.matrix(validate),
                            s = elastic_net_model1$lambda)

# 100 models (lambdas)
dim(valid.hat)

# Compute the R2 of each model to select the correct lambdas:
validate1_R2 <- apply(valid.hat,2,function(x){R2(x,validate_target)}) # For each of the 100 values, the rsquared is calculated
validate1_R2_gu <- apply(valid.hat,2,function(x){R2_gu(x,validate_target)})

plot(validate1_R2)
plot(validate1_R2_gu)

enet1_lambda <- which.max(validate1_R2)
enet1_lambda

enet1_opt_lambda <- elastic_net_model1$lambda[enet1_lambda]
enet1_opt_lambda

# Make final predictions:

# Combine training and validation samples:
train1_final <- rbind(train,validate)
target1_final <- rbind(train_target,validate_target)
dim(train1_final)
dim(target1_final)

enet1_final <- glmnet(train1_final,target1_final,
                          alpha=0,
                          lambda=enet1_opt_lambda,
                          standardize = F,
                          thresh=1e-16)

enet1_final
summary(enet1_final)
str(enet1_final)

enet1_pred <- predict.glmnet(enet1_final, newx=as.matrix(test), s=enet1_opt_lambda)

dim(enet1_pred)

enet1_final_R2 <- apply(enet1_pred,2,function(x){R2(x,test_target)})
enet1_final_R2

enet1_mse <- mse(test_target, enet1_final_R2)
enet1_mse

# Overfitting lambda
enet1_lambda_overfit <- which.min(elastic_net_model1$lambda) 

enet1_bad_lambda <- elastic_net_model1$lambda[enet1_lambda_overfit] # value of lambda is the maximum rqsquared value
enet1_bad_lambda

enet1_bad <- glmnet(train1_final,target1_final,
                        alpha=0.6,
                        lambda=enet1_bad_lambda,
                        standardize = F,
                        thresh=1e-16)

enet1_bad
summary(enet1_bad)
str(enet1_bad)


# Finally, make our final predicitons in the test sample:
enet1_final_bad <- predict.glmnet(enet1_bad,
                               newx=as.matrix(test),
                               s = enet1_bad_lambda)

names(enet1_final_bad) <- c("OverFit")

# Calculate the R2 of each model to select the correct lambdas:
enet1_final_badR2<- apply(enet1_final_bad,2,function(x){R2(x,test_target)})
enet1_final_R2
enet1_final_badR2

elastic_net_mse2 <- mse(test_target, enet1_final_bad)
elastic_net_mse2

# Elastic Net Coefficients
e_net_coefficients1 <- as.matrix(coef(enet1_final, s = "lambda.min"))
e_net_coefficients2 <- as.matrix(coef(enet1_bad, s = "lambda.min"))
e_net_coefficients1
e_net_coefficients2

selected_features1 <- rownames(as.matrix(e_net_coefficients1))[e_net_coefficients1 != 0]
selected_features2 <- rownames(as.matrix(e_net_coefficients2))[e_net_coefficients2 != 0]

cat("Selected Features:", selected_features1, "\n")
cat("Selected Features:", selected_features2, "\n")

####################################################################################################

