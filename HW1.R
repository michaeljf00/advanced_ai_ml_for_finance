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

# Train-validation-test split
train_size <- floor(0.7 * nrow(standardized))
validate_size <- floor(0.3 * nrow(standardized))

train_data <- standardized[1:train_size, ]
#validate_data <- standardized[(train_size + 1):(train_size + validate_size), ]
test_data <- standardized[(train_size + 1):nrow(standardized), ]

# Define features and target variables
features <- colnames(standardized)[1:11]
target <- "SPY"

train <- train_data[, features]
validate <- validate_data[, features]
test <- test_data[, features]

train_target <- train_data[, target]
validate_target <- validate_data[, target]
test_target <- test_data[, target]

# Model implementation
grid1 <- 10^seq(10, -2, length = 100)
grid2 <- 10^seq(15, 3, length = 100)

# Function to calculate R-squared
rsquared <- function(target, predicted) {
  1 - sum((as.vector(target) - as.vector(predicted))^2) / sum((as.vector(target) - mean(as.vector(target)))^2)
}

# Function to calculate Mean Squared Error (MSE)
mse <- function(target, predicted) {
  mean((as.vector(target) - as.vector(predicted))^2)
}

# Ridge Regression
ridge_model1 <- cv.glmnet(x = as.matrix(train), y = as.vector(train_target), alpha = 0, lambda = grid1)
best_lambda_ridge1 <- ridge_model1$lambda.min

ridge_pred1 <- predict(ridge_model1, newx = as.matrix(test), s = best_lambda_ridge1)
ridge_rsquared1 <- rsquared(test_target, ridge_pred1)
ridge_mse1 <- mse(test_target, ridge_pred1)

ridge_model2 <- cv.glmnet(x = as.matrix(train), y = as.vector(train_target), alpha = 0, lambda = grid2)
best_lambda_ridge2 <- ridge_model2$lambda.min

ridge_pred2 <- predict(ridge_model2, newx = as.matrix(test), s = best_lambda_ridge2)
ridge_rsquared2 <- rsquared(test_target, ridge_pred2)
ridge_mse2 <- mse(test_target, ridge_pred2)

# Lasso Coefficients
ridge_coefficients1 <- as.matrix(coef(ridge_model1, s = "lambda.min"))
ridge_coefficients2 <- as.matrix(coef(ridge_model2, s = "lambda.min"))

selected_features1 <- rownames(ridge_coefficients1)[ridge_coefficients1 != 0]
selected_features2 <- rownames(ridge_coefficients2)[ridge_coefficients2 != 0]

cat("Selected Features:", selected_features1, "\n")
cat("Selected Features:", selected_features2, "\n")

# Lasso Regression
lasso_model1 <- cv.glmnet(x = as.matrix(train), y = as.vector(train_target), alpha = 1, lambda = grid1)
best_lambda_lasso1 <- lasso_model1$lambda.min

lasso_pred1 <- predict(lasso_model1, newx = as.matrix(test), s = best_lambda_lasso1)
lasso_rsquared1 <- rsquared(test_target, lasso_pred1)
lasso_mse1 <- mse(test_target, lasso_pred1)

lasso_model2 <- cv.glmnet(x = as.matrix(train), y = as.vector(train_target), alpha = 1, lambda = grid2)
best_lambda_lasso2 <- lasso_model2$lambda.min

lasso_pred2 <- predict(lasso_model2, newx = as.matrix(test), s = best_lambda_lasso2)
lasso_rsquared2 <- rsquared(test_target, lasso_pred2)
lasso_mse2 <- mse(test_target, lasso_pred2)

# Lasso Coefficients
lasso_coefficients1 <- as.matrix(coef(lasso_model1, s = "lambda.min"))
lasso_coefficients2 <- as.matrix(coef(lasso_model2, s = "lambda.min"))

selected_features1 <- rownames(lasso_coefficients1)[lasso_coefficients1 != 0]
selected_features2 <- rownames(lasso_coefficients2)[lasso_coefficients2 != 0]

cat("Selected Features:", selected_features1, "\n")
cat("Selected Features:", selected_features2, "\n")


# Elastic Net Regression
elastic_net_model1 <- cv.glmnet(x = as.matrix(train), y = as.vector(train_target), alpha = 0.6, lambda = grid1)
best_lambda_elastic_net1 <- elastic_net_model1$lambda.min

elastic_net_pred1 <- predict(elastic_net_model1, newx = as.matrix(test), s = best_lambda_elastic_net1)
elastic_net_rsquared1 <- rsquared(test_target, elastic_net_pred1)
elastic_net_mse1 <- mse(test_target, elastic_net_pred1)

elastic_net_model2 <- cv.glmnet(x = as.matrix(train), y = as.vector(train_target), alpha = 0.6, lambda = grid2)
best_lambda_elastic_net2 <- elastic_net_model2$lambda.min

elastic_net_pred2 <- predict(elastic_net_model2, newx = as.matrix(test), s = best_lambda_elastic_net2)
elastic_net_rsquared2 <- rsquared(test_target, elastic_net_pred2)
elastic_net_mse2 <- mse(test_target, elastic_net_pred2)

# Elastic Net Coefficients
e_net_coefficients1 <- as.matrix(coef(elastic_net_model1, s = "lambda.min"))
e_net_coefficients2 <- as.matrix(coef(elastic_net_model2, s = "lambda.min"))

selected_features1 <- rownames(e_net_coefficients1)[e_net_coefficients1 != 0]
selected_features2 <- rownames(e_net_coefficients2)[e_net_coefficients2 != 0]

cat("Selected Features:", selected_features1, "\n")
cat("Selected Features:", selected_features2, "\n")

# Results
cat("R-squared for Ridge Regression 1:", ridge_rsquared1, "\n")
cat("MSE for Ridge Regression:", ridge_mse1, "\n")

cat("R-squared for Ridge Regression 2:", ridge_rsquared2, "\n")
cat("MSE for Ridge Regression:", ridge_mse2, "\n")

cat("R-squared for Lasso Regression 1:", lasso_rsquared1, "\n")
cat("MSE for Lasso Regression 1:", lasso_mse1, "\n")

cat("R-squared for Lasso Regression 2:", lasso_rsquared2, "\n")
cat("MSE for Lasso Regression 2:", lasso_mse2, "\n")

cat("R-squared for Elastic Net Regression 1:", elastic_net_rsquared1, "\n")
cat("MSE for Elastic Net Regression 1:", elastic_net_mse1, "\n")

cat("R-squared for Elastic Net Regression 2:", elastic_net_rsquared2, "\n")
cat("MSE for Elastic Net Regression 2:", elastic_net_mse2, "\n")

