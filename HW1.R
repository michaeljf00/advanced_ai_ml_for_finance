rm(list=ls())
library("quantmod")
library("PerformanceAnalytics")
library("glmnet")
library("caret")

set.seed(19740284)

# Get S&P 500 data
t1 = "1980-01-01"
getSymbols("SPY", from = "1980-01-01", to = Sys.Date(), adjust = TRUE)
#View(SPY)
#head(SPY, 10)

# Assign outcome variable as next period returns
volume = SPY[,5]
names(volume) <- c("SPY_vol")
adjusted = SPY[,6]
names(adjusted) <-c("SPY")
next_period_returns <- na.omit(Return.calculate(adjusted))

# Retrieve and engineer features
indeces <- c("UNRATE", "DFF", "CORESTICKM159SFRBATL", "INDPRO", "BAMLH0A0HYM2", "FEDFUNDS", "MEDCPIM158SFRBCLE", "DEXUSEU")
ind.list <- lapply(indeces, function(tic) get(getSymbols(tic, from = "2000-01-01", src = "FRED")))
IND <- na.omit(Reduce(merge,ind.list))

# 1 week rolling average
days <- 7
return_roll <- rollapply(next_period_returns, days, mean)
names(return_roll) <- paste("SPY", "_roll_", days, sep="")

# Lag
SPY_lag <- lag(next_period_returns$SPY, 1)
names(SPY_lag) <- "SPY_lag"

df_np_returns <- na.omit(merge(IND, volume, return_roll, SPY_lag, next_period_returns))
df_np_returns

# Data Pre-processing
sum(is.na(df_np_returns))

std_vars <- matrix(rep(apply(df_np_returns, 2, sd), nrow(df_np_returns)), ncol = ncol(df_np_returns), byrow = TRUE)
standardized <- df_np_returns / std_vars

# Train-test split
train_size <- floor(0.8 * nrow(standardized))
validate_size <- floor(0.2 * nrow(standardized))

train_data <- standardized[1:train_size,]
#validate_data <- df_np_returns[(train_size+1):(train_size+validate_size), ]
test_data <- standardized[(train_size+1):nrow(standardized),]

train <- train_data[, 1:11]
#validate <- validate_data[,1:11]
test <- test_data[, 1:11]

train_target <- train_data[,c("SPY")]
#validate_target <- validate_data[,c("SPY")]
test_target <- test_data[,c("SPY")]

# Model implementation
grid1 <- 10^seq(10,-2,length=100)
grid2 <- 10^seq(15, 3, length=100)

MSE <- function(prediction, target) {
  return(mean((as.vector(prediction) - as.vector(target))^2))
}

rsquared <- function(target, predicted) {
  return(1 - (sum((as.vector(target) - as.vector(predicted))^2) / sum((as.vector(target) - mean(as.vector(target)))^2)))
}

# Ridge Regression
ridge1_model <- glmnet(train, train_target, alpha = 0, lambda = grid1)
ridge2_model <- glmnet(train, train_target, alpha = 0, lambda = grid2)

ridge1_pred <- predict(ridge1_model, newx <- as.matrix(test))
ridge2_pred <- predict(ridge2_model, newx <- as.matrix(test))

ridge1_rsquared <- rsquared(ridge1_pred, test_target)
ridge2_rsquared <- rsquared(ridge2_pred, test_target)

ridge1_mse <- MSE(ridge1_pred, test_target)
ridge2_mse <- MSE(ridge2_pred, test_target)

# Lasso Regression
lasso1_model <- glmnet(train, train_target, alpha = 1, lambda = grid1, standardize = F)
lasso2_model <- glmnet(train, train_target, alpha = 1, lambda = grid2, standardize = F)

lasso1_pred <- predict(lasso1_model, newx <- as.matrix(test))
lasso2_pred <- predict(lasso2_model, newx <- as.matrix(test))

lasso1_rsquared <- rsquared(lasso1_pred, test_target)
lasso2_rsquared <- rsquared(lasso2_pred, test_target)

lasso1_mse <- MSE(lasso1_pred, test_target)
lasso2_mse <- MSE(lasso2_pred, test_target)

# Elastic Net Regression
elastic_net1_model <- glmnet(train, train_target, alpha = 0.5, lambda = grid1, standardize = F)
elastic_net2_model <- glmnet(train, train_target, alpha = 0.5, lambda = grid2, standardize = F)

elastic_net1_pred <- predict(elastic_net1_model, newx <- as.matrix(test))
elastic_net2_pred <- predict(elastic_net2_model, newx <- as.matrix(test))

elastic_net1_rsquared <- rsquared(elastic_net1_pred, test_target)
elastic_net2_rsquared <- rsquared(elastic_net2_pred, test_target)

elastic_net1_mse <- MSE(elastic_net1_pred, test_target)
elastic_net2_mse <- MSE(elastic_net2_pred, test_target)

# Results

ridge1_rsquared
ridge2_rsquared
ridge1_mse
ridge2_mse

lasso1_rsquared
lasso2_rsquared
lasso1_mse
lasso2_mse

elastic_net1_rsquared
elastic_net2_rsquared
elastic_net1_mse
elastic_net2_mse

nrow(test)
nrow(train)
