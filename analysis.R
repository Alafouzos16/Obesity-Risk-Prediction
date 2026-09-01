#===============================================================================
# Data and library importing, data summary
#===============================================================================

library(here)
library(dplyr)
library(labelled)
library(car)
library(gtsummary)
library(flextable)
library(pROC)
library(glmnet)
library(randomForest)
library(caret)
library(DALEX)
library(xgboost)
library(nnet)
library(forcats)

data <- read.csv(here("data", "ObesityDataSet.csv"), header = TRUE, sep = ',')
dir.create(here("output"), showWarnings = FALSE)

str(data)

#===============================================================================
# Character variables to factors, cleaning data, combining categories 
# with 0/few observations
#===============================================================================

data <- data %>%
  mutate(
    across(c(Gender, family_history_with_overweight, FAVC, SMOKE, SCC), as.factor),
    
    CAEC = factor(CAEC, levels = c('no', 'Sometimes', 'Frequently', 'Always')),
    CALC = fct_collapse(CALC,
                        no = "no",
                        Sometimes = "Sometimes",
                        Frequently_or_Always = c("Frequently", "Always")),
    
    MTRANS = fct_collapse(MTRANS,
                          Automobile = "Automobile",
                          Public_Transportation = "Public_Transportation",
                          Other = c("Bike", "Motorbike", "Walking")),
    
    NObeyesdad = factor(NObeyesdad, levels = c('Insufficient_Weight', 'Normal_Weight', 
                                               'Overweight_Level_I', 'Overweight_Level_II', 
                                               'Obesity_Type_I', 'Obesity_Type_II', 
                                               'Obesity_Type_III')),
    
    FCVC_category = factor(round(FCVC), 
                           levels = c(1, 2, 3), 
                           labels = c("Never", "Sometimes", "Always")),
    
    NCP_category = case_when(
      round(NCP) <= 2 ~ "Between 1 & 2",
      round(NCP) >= 3 ~ "Three or more"
    ),
    NCP_category = factor(NCP_category, levels = c("Between 1 & 2", "Three or more")),
    
    CH2O_category = factor(round(CH2O), 
                           levels = c(1, 2, 3), 
                           labels = c("Less than 1 L", "Between 1 and 2 L", "More than 2 L")),
    
    FAF_category = factor(round(FAF), 
                          levels = c(0, 1, 2, 3), 
                          labels = c("No physical activity", "1 or 2 days", "2 to 4 days", "4 or 5 days")),
    
    TUE_category = factor(round(TUE), 
                          levels = c(0, 1, 2), 
                          labels = c("0 to 2 hours", "3 to 5 hours", "More than 5 hours"))
  )

str(data)

#===============================================================================
# Creating new and correct dataset
#===============================================================================

# Creating data2 excluding Height, Weight and auxiliary variables
data2 <- data %>%
  select(Gender, Age, family_history_with_overweight, FAVC, FCVC_category, 
         NCP_category, CH2O_category, FAF_category, TUE_category, CAEC, 
         SMOKE, SCC, CALC, MTRANS, NObeyesdad) %>%
  mutate(
    # Setting "Normal_Weight" as reference category for Multinomial Regression
    NObeyesdad = relevel(NObeyesdad, ref = "Normal_Weight"),
    
    # Creating Is_Obese variable for GLM
    Is_Obese = case_when(
      NObeyesdad %in% c("Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II") ~ "No",
      NObeyesdad %in% c("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III") ~ "Yes"
    ),
    # Setting "No" as reference category
    Is_Obese = factor(Is_Obese, levels = c("No", "Yes"))
  )

str(data2)

#===============================================================================
# Labeling variables
#===============================================================================

var_label(data2) <- list(
  Gender = "Gender",
  Age = "Age (in years)",
  family_history_with_overweight = "Family History with Overweight Relative",
  FAVC = "Frequent consumption of high caloric food",
  FCVC_category = "Frequency of consumption of vegetables",
  NCP_category = "Number of Main Meals",
  CAEC = "Consumption of food between meals",
  SMOKE = "Smoking",
  CH2O_category = "Consumption of water daily",
  SCC = "Calories consumption monitoring",
  FAF_category = "Physical activity frequency per week",
  TUE_category = "Hours using technology devices per day",
  CALC = "Consumption of alcohol",
  MTRANS = "Transportation of use",
  NObeyesdad = "Obesity Level",
  Is_Obese = "Being Obese"
)

var_label(data2)

#===============================================================================
# Descriptive statistics
#===============================================================================

sapply(data2[, sapply(data2, is.numeric), drop = FALSE], function(x) c(
  n = sum(!is.na(x)),
  missing = sum(is.na(x)),
  mean = mean(x, na.rm = TRUE),
  sd = sd(x, na.rm = TRUE),
  median = median(x, na.rm = TRUE),
  IQR = IQR(x, na.rm = TRUE),
  min = min(x, na.rm = TRUE),
  max = max(x, na.rm = TRUE)
))

factors <- names(data2)[sapply(data2, is.factor)]
categorical_summary <- lapply(factors, function(col) {
  freq_table <- table(data2[[col]], useNA = "ifany")
  prop_table <- prop.table(freq_table) * 100
  cbind(Freq = freq_table, Percentage = round(prop_table, 2))
})
names(categorical_summary) <- factors
categorical_summary

#===============================================================================
# Binomial Regression
#===============================================================================

binary_model <- glm(Is_Obese ~ Gender + Age + family_history_with_overweight + 
                      FAVC + FCVC_category + NCP_category + CH2O_category + 
                      FAF_category + TUE_category + CAEC + SMOKE + SCC + CALC + MTRANS, 
                    data = data2, 
                    family = binomial) # including all variables and not overlapping

summary(binary_model)

Anova(binary_model, type = "II")

# excluding not significant variables one at a time

binary_model_2 <- update(binary_model, . ~ . - TUE_category)
Anova(binary_model_2, type = "II")

binary_model_3 <- update(binary_model_2, . ~ . - Gender)
Anova(binary_model_3, type = "II")

binary_model_4 <- update(binary_model_3, . ~ . - SMOKE)
Anova(binary_model_4, type = "II")

binary_model_final <- update(binary_model_4, . ~ . - CALC)
Anova(binary_model_final, type = "II")

#===============================================================================
# Log Age and interactions
#===============================================================================

data2 <- data2 %>%
  mutate(logAge = log(Age))
var_label(data2$logAge) <- "Log Age"

# model_linear is identical to binary_model_final — kept as a named baseline
# for the AIC comparison against model_log below
model_linear <- binary_model_final

# Replacing Age with logAge
model_log <- update(model_linear, . ~ . - Age + logAge)

# Comparing AIC
AIC(model_linear, model_log)

# Adding family_history_with_overweight * FAVC interaction
model_inter <- update(model_log, . ~ . + family_history_with_overweight:FAVC)
Anova(model_inter, type = "II")
AIC(model_inter, model_log)

# Adding FAF_category * CH2O_category interaction
model_inter2 <- update(model_inter, . ~ . + FAF_category:CH2O_category)
Anova(model_inter2, type = "II")
AIC(model_inter, model_inter2)

#===============================================================================
# Multicollinearity check for model_inter2 (workaround for glm + interactions)
#===============================================================================

# car::vif(type="predictor") requires an lm object; model_inter2 is a glm (binomial).
# As before, VIF depends only on the predictor design matrix, not on the response
# type, so we fit an auxiliary lm() with the same RHS to obtain valid VIF values.

vif_check_binomial <- lm(as.numeric(Is_Obese) ~ family_history_with_overweight + 
                           FAVC + FCVC_category + NCP_category + CH2O_category + 
                           FAF_category + CAEC + SCC + MTRANS + logAge + 
                           family_history_with_overweight:FAVC + FAF_category:CH2O_category,
                         data = data2)

vif(vif_check_binomial, type = "predictor")

model_final <- model_inter2

#===============================================================================
# Odds Ratio of Binomial Model
#===============================================================================

final_table <- tbl_regression(model_final, exponentiate = TRUE) %>%
  add_global_p() %>%
  bold_p(t = 0.05)

final_table

final_table %>%
  as_flex_table() %>%
  save_as_docx(path = here("output", "Binomial_Results.docx"))

#===============================================================================
# ROC curve and AUC
#===============================================================================

# prediction for each participant to be obese or not
predicted_probs <- predict(model_final, type = "response")

# ROC Curve for real obese people vs predictions
roc_obj <- roc(data2$Is_Obese, predicted_probs)

png(here("output", "ROC_Binomial_Model.png"), width = 800, height = 600, res = 120)

# ROC curve visualisation
plot(roc_obj, 
     main = "ROC Curve: Final Model", 
     col = "#1b9e77",
     lwd = 3,
     print.auc = TRUE,
     print.auc.x = 0.5,
     print.auc.y = 0.3, 
     legacy.axes = TRUE)

dev.off()

#===============================================================================
# Machine Learning with Lasso
#===============================================================================

# Splitting data set into train and test set, keeping Is_Obese as the desirable variable
set.seed(123)
train_index <- createDataPartition(data2$Is_Obese, p = 0.7, list = FALSE)

train_data <- data2[train_index, ]
test_data <- data2[-train_index, ]

# Preparing Data — using train_data only, to avoid data leakage
x_train <- model.matrix(Is_Obese ~ logAge + family_history_with_overweight * FAVC + 
                          FCVC_category + NCP_category + CAEC + SCC + MTRANS + 
                          FAF_category * CH2O_category, data = train_data)[, -1]
y_train <- ifelse(train_data$Is_Obese == "Yes", 1, 0)

x_test <- model.matrix(Is_Obese ~ logAge + family_history_with_overweight * FAVC + 
                         FCVC_category + NCP_category + CAEC + SCC + MTRANS + 
                         FAF_category * CH2O_category, data = test_data)[, -1]
y_test <- ifelse(test_data$Is_Obese == "Yes", 1, 0)

# Cross-Validation (on training data only)
set.seed(123)
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")

# Penalty plot
plot(lasso_cv)

lasso_coefficients <- coef(lasso_cv, s = "lambda.1se")
lasso_coefficients

# LASSO Predictions — on held-out test set
lasso_probs <- predict(lasso_cv, newx = x_test, s = "lambda.1se", type = "response")

# Making them numeric
lasso_probs <- as.numeric(lasso_probs)

lasso_roc <- roc(y_test, lasso_probs)

png(here("output", "ROC_LASSO.png"), width = 800, height = 600, res = 120)

plot(lasso_roc, 
     main = "ROC Curve: LASSO Model", 
     col = "#e7298a",
     lwd = 3, 
     print.auc = TRUE, 
     legacy.axes = TRUE)

dev.off()

#===============================================================================
# Machine Learning with Random Forest
#===============================================================================

# Using Random Forest method with the significant values
rf_model <- randomForest(Is_Obese ~ logAge + family_history_with_overweight + 
                           FAVC + FCVC_category + NCP_category + CH2O_category + 
                           FAF_category + CAEC + SCC + MTRANS, 
                         data = train_data, 
                         ntree = 500,
                         importance = TRUE)

# Predicting the test set
rf_predictions <- predict(rf_model, test_data)

# Confusion Matrix
confusionMatrix(rf_predictions, test_data$Is_Obese)

png(here("output", "RandomForest_Variable_Importance.png"), width = 800, height = 600, res = 120)

varImpPlot(rf_model)

dev.off()

# Random Forest probabilities
rf_probs <- predict(rf_model, test_data, type = "prob")[, "Yes"]

# ROC Curve for Random Forest
rf_roc <- roc(test_data$Is_Obese, rf_probs, levels = c("No", "Yes"))

png(here("output", "ROC_RandomForest.png"), width = 800, height = 600, res = 120)

plot(rf_roc, 
     main = "ROC Curve: Random Forest Model", 
     col = "#ff7f00",
     lwd = 3, 
     print.auc = TRUE, 
     legacy.axes = TRUE)

dev.off()

#===============================================================================
# Random Forest Hyperparameter Tuning
#===============================================================================

# 10-Fold CV
fitControl <- trainControl(method = "cv", 
                           number = 10, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)

# Grid of values
tuneGrid <- expand.grid(.mtry = c(2, 3, 4, 5, 6, 7))

# Finding optimal mtry
set.seed(123)
rf_tuned <- train(Is_Obese ~ logAge + family_history_with_overweight + 
                    FAVC + FCVC_category + NCP_category + CH2O_category + 
                    FAF_category + CAEC + SCC + MTRANS, 
                  data = train_data, 
                  method = "rf", 
                  metric = "ROC",
                  tuneGrid = tuneGrid, 
                  trControl = fitControl,
                  ntree = 500)

# Results visualisation
print(rf_tuned)
plot(rf_tuned, main = "AUC result based on mtry")

#===============================================================================
# Youden's Index
#===============================================================================

# New probabilities for Test Set with optimal model
tuned_probs <- predict(rf_tuned, test_data, type = "prob")[, "Yes"]

# ROC curve for new model
tuned_roc <- roc(test_data$Is_Obese, tuned_probs, levels = c("No", "Yes"))

# Ideal cutoff with Youden's Index
coords(tuned_roc, "best", ret = c("threshold", "specificity", "sensitivity"), best.method = "youden")

# 55.8% probability is optimal to determine obesity

#===============================================================================
# SHAP Values with DALEX
#===============================================================================

# Isolating variables
features <- c("logAge", "family_history_with_overweight", "FAVC", "FCVC_category", 
              "NCP_category", "CH2O_category", "FAF_category", "CAEC", "SCC", "MTRANS")

# Creating Explainer
explainer_rf <- DALEX::explain(model = rf_tuned, 
                        data = train_data[, features], 
                        y = as.numeric(train_data$Is_Obese == "Yes"),
                        predict_function = function(m, newdata) predict(m, newdata, type = "prob")[, "Yes"],
                        label = "Random Forest (Tuned)")

# Selecting Patient No.1 from test set
patient_1 <- test_data[1, features]

# SHAP value calculation for this patient
shap_patient <- predict_parts(explainer_rf, 
                              new_observation = patient_1, 
                              type = "shap")

png(here("output", "SHAP_Patient_No_1.png"), width = 800, height = 600, res = 120)

# SHAP plot
plot(shap_patient)

dev.off()

#===============================================================================
# LIME Explanations
#===============================================================================

library(lime)

# library(lime) is loaded here (not in the top block) so that the SHAP
# section above still resolves the unqualified explain() to DALEX::explain().
# Both `lime` and `DALEX` export a function called `explain()`.
# Since DALEX is loaded first in this script, `explain()` unqualified
# would resolve to DALEX's version — we use `lime::explain()` explicitly
# to avoid silently calling the wrong function.

# Creating LIME explainer on the same features/data used for SHAP
lime_explainer <- lime(
  x = train_data[, features],
  model = rf_tuned
)

# Explaining the same Patient No.1 used in the SHAP section, for direct comparison
lime_explanation <- lime::explain(
  x = patient_1,
  explainer = lime_explainer,
  n_labels = 1,
  n_features = 10,
  n_permutations = 5000,
  kernel_width = 0.5
)

png(here("output", "LIME_Patient_No_1.png"), width = 800, height = 600, res = 120)

# LIME plot
plot_features(lime_explanation)

dev.off()

#===============================================================================
# XGBoost Model
#===============================================================================

train_data$Is_Obese <- as.factor(train_data$Is_Obese)
test_data$Is_Obese <- as.factor(test_data$Is_Obese)

dummies <- dummyVars(~ logAge + family_history_with_overweight + 
                       FAVC + FCVC_category + NCP_category + CH2O_category + 
                       FAF_category + CAEC + SCC + MTRANS, 
                     data = train_data)

train_data_xgb <- data.frame(predict(dummies, newdata = train_data))
test_data_xgb <- data.frame(predict(dummies, newdata = test_data))

train_data_xgb$Is_Obese <- train_data$Is_Obese
test_data_xgb$Is_Obese <- test_data$Is_Obese

colnames(train_data_xgb) <- make.names(colnames(train_data_xgb))
colnames(test_data_xgb) <- make.names(colnames(test_data_xgb))

stopifnot(identical(setdiff(names(train_data_xgb), "Is_Obese"),
                    setdiff(names(test_data_xgb), "Is_Obese")))

dtrain <- xgb.DMatrix(data = as.matrix(train_data_xgb[, setdiff(names(train_data_xgb), "Is_Obese")]),
                      label = as.numeric(train_data_xgb$Is_Obese == "Yes"))

params <- list(objective = "binary:logistic", eval_metric = "auc",
               max_depth = 3, eta = 0.3, gamma = 0,
               colsample_bytree = 1, min_child_weight = 1, subsample = 1)

set.seed(123)
xgb_cv <- xgb.cv(params = params, data = dtrain, nrounds = 100,
                 nfold = 10, showsd = TRUE, stratified = TRUE,
                 print_every_n = 10, early_stopping_rounds = 10)

best_nrounds <- which.max(xgb_cv$evaluation_log$test_auc_mean)

set.seed(123)
xgb_final <- xgb.train(params = params, data = dtrain, nrounds = best_nrounds)

dtest <- xgb.DMatrix(data = as.matrix(test_data_xgb[, setdiff(names(test_data_xgb), "Is_Obese")]))
xgb_probs <- predict(xgb_final, dtest)

xgb_roc <- roc(test_data_xgb$Is_Obese, xgb_probs, levels = c("No", "Yes"))

png(here("output", "ROC_XGBoost.png"), width = 800, height = 600, res = 120)

plot(xgb_roc, main = "ROC Curve: XGBoost Model", col = "#e41a1c", lwd = 3, print.auc = TRUE, legacy.axes = TRUE)

dev.off()

#===============================================================================
# Multinomial Regression
#===============================================================================

# NObeyesdad is already releveled with "Normal_Weight" as reference (set in data2)
multinom_model <- multinom(NObeyesdad ~ Gender + Age + family_history_with_overweight + 
                             FAVC + FCVC_category + NCP_category + CH2O_category + 
                             FAF_category + TUE_category + CAEC + SMOKE + SCC + CALC + MTRANS, 
                           data = data2,
                           trace = FALSE) # trace = FALSE suppresses iteration log

summary(multinom_model)

# Type II Wald tests for each predictor across all category comparisons
Anova(multinom_model, type = "II")

#===============================================================================
# Log Age and interactions (Multinomial)
#===============================================================================

multinom_model_log <- update(multinom_model, . ~ . - Age + logAge)
AIC(multinom_model, multinom_model_log)

# Adding family_history_with_overweight * FAVC interaction
multinom_model_inter <- update(multinom_model_log, . ~ . + family_history_with_overweight:FAVC)
Anova(multinom_model_inter, type = "II")
AIC(multinom_model_inter, multinom_model_log)

# Adding FAF_category * CH2O_category interaction
multinom_model_inter2 <- update(multinom_model_inter, . ~ . + FAF_category:CH2O_category)
Anova(multinom_model_inter2, type = "II")
AIC(multinom_model_inter, multinom_model_inter2)

#===============================================================================
# Multicollinearity check for Multinomial Model (workaround)
#===============================================================================

# car::vif() does not support nnet::multinom objects directly (returns NaN)
# because of how their combined vcov() matrix is structured across classes.
# VIF depends only on the predictor design matrix, not on the response type,
# so we fit an auxiliary lm() with the same RHS to obtain valid VIF values.

vif_check_multinom <- lm(as.numeric(NObeyesdad) ~ Gender + family_history_with_overweight + 
                           FAVC + FCVC_category + NCP_category + CH2O_category + 
                           FAF_category + TUE_category + CAEC + SMOKE + SCC + CALC + MTRANS + 
                           logAge + family_history_with_overweight:FAVC,
                         data = data2)

vif(vif_check_multinom, type = "predictor")

multinom_model_final <- multinom_model_inter

#===============================================================================
# Odds Ratio of Multinomial Model
#===============================================================================

multinom_table <- tbl_regression(multinom_model_final, exponentiate = TRUE) %>%
  add_global_p() %>%
  bold_p(t = 0.05)

multinom_table

multinom_table %>%
  as_flex_table() %>%
  save_as_docx(path = here("output", "Multinomial_Results.docx"))

#===============================================================================
# Note on model limitations
#===============================================================================

# The multinomial model shows unstable (very wide CI / extreme OR) estimates
# for the Obesity_Type_II and especially Obesity_Type_III classes. This is not
# a data cleaning artifact — it reflects genuine near-complete separation:
# these classes are highly homogeneous across several predictors simultaneously
# (e.g., virtually all Obesity_Type_III cases share NCP_category = "Three or
# more" and MTRANS = "Public_Transportation"), which prevents stable maximum
# likelihood estimation. Bias-reduced estimation (brglm2::brmultinom, both the
# iterative "AS_mean" and one-shot "correction" methods) was attempted and
# failed to resolve this, confirming the issue is structural rather than a
# numerical/computational limitation. Odds ratios for these two classes should
# be interpreted with caution: direction of association is broadly informative,
# magnitude and CI width are not reliable point estimates.