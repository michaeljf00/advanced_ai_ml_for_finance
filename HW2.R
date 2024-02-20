rm(list=ls()) # clear the memory	
setwd("/Users/michaeljoshua/Desktop/Advanced AI_ML for Finance")	
library("ggplot2")	
library("reshape")	
library("plm")	
library("rpart")	
library("zoo")	
library("plyr")	
library("dplyr")	
library("stringr")	
library("reshape2")	
library("ggplot2")	
library("pander")	
library("DataCombine")	
library("plm")	
library("quantmod")	
library("caret")
library("ISLR")
library("e1071")
library("ipred")

set.seed(187420298)

# Import the mortgage data:	
load("Mortgage_Annual.Rda")	
ncol(down_train)
nrow(down_train)
# Rename the data
df <- p.mort.dat.annual
rm(p.mort.dat.annual)

df <- pdata.frame(df, index=c("LOAN_ID", "year"), stringsAsFactors = F)

# Print the class of the variable:
class(df)

# Generate Variables we want
# 1. Default 1/0 indicator (180+ DPD):
df$def <- 0
# Save the indeces (rows) of
tmp <- which(df$F180_DTE == df$date)
df$def[tmp] <- 1

# 2. Replace NUM_UNIT with MULTI_UNIT dummy:
table(df$NUM_UNIT)

df$MULTI_UN <- 0
tmp <- which(df$NUM_UNIT > 1)
df$MULTI_UN[tmp] <- 1

# 3. Count the number of loans:
print(length(unique(df$LOAN_ID)))

# Compress the data to single loans:
df.annual <-df %>%
  group_by(LOAN_ID) %>%
  mutate(def.max = max(def)) %>%
  mutate(n = row_number()) %>%
  ungroup()

# Print the variable nammes in df.annual
names(df.annual)

# keep one obs per loan:
tmp <- which(df.annual$n == 1)
df.annual <- df.annual[tmp, ]
dim(df.annual)
df.annual
# Keep only relevant variables for default analysis:
my.vars <- c("ORIG_CHN","ORIG_RT",
             "ORIG_AMT","ORIG_TRM","OLTV",
             "DTI","OCC_STAT",
             "MULTI_UN",
             "CSCORE_MN",
             "ORIG_VAL",
             "VinYr","def.max")
df.model <- subset(df.annual, select=my.vars)
names(df.model)

table(df.model$def.max)

tmp <- table(df.model$def.max)
df.rate <- tmp[2] / sum(tmp) * 100
message(sprintf("The default rate is %4.2f%%", df.rate))

# Print the objects in memory:
ls()

# Remove all but df.model
rm(list=setdiff(ls(), "df.model"))
ls()

head(df.model)

df.model.noNA <- df.model[complete.cases(df.model),]
df.model.noNA$def.max <- as.factor(df.model.noNA$def.max)

# Downsample:
down_train <- downSample(x = df.model.noNA[, -ncol(df.model.noNA)],
                         y = df.model.noNA$def.max)

down_train
# Split into train/test:

# Function to calculate model accuracy
calculate_accuracy <- function(predictions, actual) {
  mean(predictions == actual)
}

# 2a - 10 random splits
accuracy_values <- numeric()

for (i in 1:10) {
  splitIndex <- createDataPartition(down_train$Class, p = 0.7, list = FALSE)
  train_data <- down_train[splitIndex, ]
  test_data <- down_train[-splitIndex, ]

  # Fit SVM model
  svm_model <- svm(Class~ ., data = train_data, kernel = "radial", cost = 1)
  
  # Predict 
  svm_pred <- predict(svm_model, newdata = test_data[,-ncol(test_data)])
  
  # Accuracy
  accuracy_values[i] <- calculate_accuracy(svm_pred, test_data$Class)
}

cat("Mean Accuracy across 10 splits:", mean(accuracy_values), "\n")

# 2b LOOCV
ctrl <- trainControl(method = "LOOCV")

svm_loocv <- train(Class ~ ., 
                   data = down_train, 
                   method = "svmRadial", 
                   trControl = ctrl)

loocv_results <- svm_loocv$results

mean_accuracy_loocv <- mean(loocv_results$Accuracy)
cat("LOOCV Model Accuracy:", mean_accuracy_loocv, "\n")


# 2c - k-Fold Cross-Validation
# K-fold cross-validation (k = 2 to 10)
library("kernlab")

kfold_accuracies <- sapply(2:10, function(k) {
  set.seed(187420298)  # Set seed for reproducibility in each iteration
  cv_results <- train(
    Class~ ., 
    data = down_train, 
    method = "svmRadial", 
    trControl = trainControl(method = "cv", number = k)
  )
  return(cv_results$results$Accuracy)
})

cat("K-fold Cross-Validation Accuracies:", kfold_accuracies, "\n")
cat("Overal Model Accuracy:", mean(kfold_accuracies), "\n")

# 3
library("ipred")

ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# 3a - Fit bagged tree model
bag_model <- train(Class ~ ., data = down_train, method = "treebag", trControl = ctrl)
print(bag_model)

# 3b - Random Forest
rf_grid <- expand.grid(mtry = seq(1, ncol(down_train)-1))
rf_model <- train(Class ~ ., data = down_train, method = "rf", trControl = ctrl, tuneGrid = rf_grid)
print(rf_model)

# 3c - Boosted Tree (Using xgboost)

boost_grid <- expand.grid(
  nrounds = seq(50, 200, by = 10),
  max_depth = c(3, 6, 9),
  eta = seq(0.01, 0.2, by = 0.01),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

boost_model <- train(
  Class ~ ., 
  data = down_train, 
  method = "xgbTree", 
  trControl = ctrl, 
  tuneGrid = boost_grid
)
print(boost_model)

