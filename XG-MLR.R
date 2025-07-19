# Program that implements two models of capacity as the resposnse variable.
# XGBoost and Multiple Linear Regression.

# Install and load required packages
install.packages("xgboost")
install.packages("Metrics")

library(xgboost)
library(Metrics)

setwd("C:/Users/Kenneth Kamogelo/OneDrive - University of Cape Town/Desktop/CSIR/Vac work/Code")

# Load data
df <- read.csv("Test_balanced_discharge_data.csv")



# Define features and target
features <- c("current_measured", "cycle")
target <- "capacity"

# Create feature matrix and target vector
X <- as.matrix(df[, features])
y <- df[[target]]

# Manual 80/20 train-test split
set.seed(42)
n <- nrow(X)
train_indices <- sample(1:n, size = 0.8 * n)

X_train <- X[train_indices, , drop = FALSE]
X_test  <- X[-train_indices, , drop = FALSE]
y_train <- y[train_indices]
y_test  <- y[-train_indices]

# Multiple Linear Regression
train_df <- as.data.frame(cbind(capacity = y_train, X_train))
test_df  <- as.data.frame(cbind(capacity = y_test, X_test))

# Fit the model using both features
mlr_model <- lm(capacity ~ cycle + current_measured, data = train_df)
summary(mlr_model)
# Predict on test set
predictions <- predict(mlr_model, newdata = test_df)

# Calculate RMSE
rmse_val <- rmse(test_df$capacity, predictions)

# Calculate R-squared manually
ss_res <- sum((test_df$capacity - predictions)^2)
ss_tot <- sum((test_df$capacity - mean(test_df$capacity))^2)
r_squared <- 1 - ss_res / ss_tot

# Print results
cat("RMSE:", round(rmse_val, 4), "\n")
cat("R-squared:", round(r_squared, 4), "\n")

plot(test_df$capacity, predictions, 
     xlab = "Actual Capacity", ylab = "Predicted Capacity",
     main = "Actual vs Predicted Capacity")
abline(0, 1, col = "red")  # Line y = x

# Predict and evaluate
mlr_preds <- predict(mlr_model, newdata = test_df)
rmse_mlr <- rmse(y_test, mlr_preds)
mae_mlr  <- mae(y_test, mlr_preds)

# xGBoost
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test)

xgb_model <- xgboost(
  data = dtrain,
  nrounds = 100,
  max_depth = 3,
  eta = 0.1,
  objective = "reg:squarederror",
  verbose = 0
)
summary(xgb_model)

xgb_preds <- predict(xgb_model, newdata = dtest)
rmse_xgb <- rmse(y_test, xgb_preds)
mae_xgb  <- mae(y_test, xgb_preds)

# Results
cat("Multiple Linear Regression:\n")
cat("  RMSE:", round(rmse_mlr, 4), "\n")
cat("  MAE :", round(mae_mlr, 4), "\n\n")

cat("XGBoost:\n")
cat("  RMSE:", round(rmse_xgb, 4), "\n")
cat("  MAE :", round(mae_xgb, 4), "\n")

