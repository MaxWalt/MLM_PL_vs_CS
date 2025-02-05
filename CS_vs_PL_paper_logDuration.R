#######################################################################################
#                                                                                     #      
#                 Using MLM to compare Power Law vs Critical speed                    #
#                 ------------------------------------------------                    #  
#                                                                                     #
#   0. Load mandatory material and prepare datasets                                   #
#   1. Cross-validation process of the MLM models                                     #
#   2. PL MLM                                                                         #
#   3. CS MLM                                                                         #
#   4. Predictions                                                                    #
#   5. Parameters description                                                         #
#   6. Correlation between models parameters and performance in events                #
#   7. Real life cases                                                                #
#                                                                                     #
#                                                       June 2024 - February 2025     #
#                                                                                     #
#######################################################################################

# READ ME: In this script we aim to compare CS and PL model using MLM structure.
#          We compare CS and PL MLM models in terms of MAE, MARE, and characterization abilities.

#=== 0. Load mandatory material and prepare datasets

# 0.1. Load the necessary libraries

library(dplyr)
library(tidyr)
library(purrr)
library(tidyverse)
library(ggplot2)
library(readxl)
library(table1)
library(caret)
library(lme4)
library(Matrix)
library(e1071)
library(metrica)
library(performance)
library(ggpubr)
library(extrafont)
library(patchwork)
library(digest)
library(ggtext)
library(outliers)
library(RColorBrewer)

# 0.2. Load the necessary files

load("C:/Users/maxim/Documents/UNIL - Doctorat/1_Research_data/2_Study_A_CS_vs_PL/CS_vs_PL/CS_vs_PL/MDLD_speed")

# MDLD_speed is a wide format df that stores each athletes PB (400-10,000m) in time (s), speed (m/s) and WA points.

# 0.3. Adjust the dataset to the needs of the study

# Transform the dataset to long format
MDLD_speed_long <- MDLD_speed %>%
  # Pivot the 'speed_' columns
  pivot_longer(
    cols = starts_with("speed_"), 
    names_to = "distance", 
    values_to = "speed", 
    names_prefix = "speed_", 
    values_drop_na = TRUE 
  ) %>%
  mutate(
    distance = as.numeric(distance) # Convert distance names to numeric
  ) %>%
  # Pivot the 'time_' columns
  pivot_longer(
    cols = starts_with("time_"),
    names_to = "distance_time",
    values_to = "time",
    names_prefix = "time_",
    values_drop_na = TRUE
  ) %>%
  mutate(
    distance_time = as.numeric(distance_time) # Convert distance_time to numeric
  ) %>%
  # Filter to align speed and time by matching distances
  filter(distance == distance_time) %>%
  dplyr::select(ID, Gender, Best_event, Max_level, distance, speed, time)

# Compute log-transformed speed and duration for the Power Law model and scale the Best_perf
MDLD_speed_long <- MDLD_speed_long %>%
  mutate(
    log_speed = log(speed),
    log_time = log(time)
  )

# 0.4. Population description

# Male

MDLD_speed_MAN <- subset(MDLD_speed, MDLD_speed$Gender=="Male")

Pop_all_M <- MDLD_speed_MAN[, c(3:9, 16:21)] 

Mean_M <- Pop_all_M %>% 
  group_by(Best_event) %>%
  summarise_all(list(mean = mean))

SD_M <- Pop_all_M %>% 
  group_by(Best_event) %>%
  summarise_all(list(sd = sd))

Pop_all_M %>% 
  group_by(Best_event) %>%
  summarise(total_count = n())

# Female 

MDLD_speed_WOM <- subset(MDLD_speed, MDLD_speed$Gender=="Female")

Pop_all_W <- MDLD_speed_WOM[, c(3:9, 16:21)]

Mean_W <- Pop_all_W %>% 
  group_by(Best_event) %>%
  summarise_all(list(mean = mean))

SD_W <- Pop_all_W %>% 
  group_by(Best_event) %>%
  summarise_all(list(sd = sd))

Pop_all_W %>% 
  group_by(Best_event) %>%
  summarise(total_count = n())

MDLD_speed %>%
  group_by(Best_event) %>%
  summarise(total_count = n())

#=== 1. Model selection process of the MLM models

# Compute various PL models (from simpler to more complex)

# According to Finch (2014) and Hox (2002), the model selection process of MLM starts with comparing 
# AIC, BIC and logLik scores between different models complexity and run an anova to compute the 
# likelihood ratio test

# For reproducibility 
set.seed(123)

# Compute NULL-model 1: only individual level (intercept only)

PL_MLM_null_1 <- lmer(log_speed ~ 1 + (1 | ID), data = MDLD_speed_long, REML = FALSE,
                      control = lmerControl(optimizer = "bobyqa"))

# Compute model 2: log-time and ID level (intercept only)

PL_MLM_2 <- lmer(log_speed ~ 1 + log_time + (1 | ID), data = MDLD_speed_long, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))

anova(PL_MLM_null_1, PL_MLM_2) # p-val < 0.05

# Compute model 3: log-time, ID level (intercept and slope)

PL_MLM_3 <- lmer(log_speed ~ 1 + log_time + (1 + log_time | ID), data = MDLD_speed_long, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))

anova(PL_MLM_2, PL_MLM_3) # p-val < 0.05

# Compute model 4: log-time, Gender and ID level (intercept and slope)

PL_MLM_4 <- lmer(log_speed ~ 1 + log_time + Gender + (1 + log_time | ID), data = MDLD_speed_long, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))

anova(PL_MLM_3, PL_MLM_4) # p-val < 0.05

# Compute model 5: log-time, Gender, ID (intercept and slope) and Best event levels (intercept)
# Equation: log(speed)ijk = gamma_{000} + b_{0j} + u_{0i} + gamma_{200} * Gender_{ij}
#                           + (gamma_{100} + u_{1i}) * log(time)_{ijk} + epsilon_{ijk}^{PL}

PL_MLM <- lmer(log_speed ~ 1 + log_time + Gender + (1 + log_time | ID) + (1 | Best_event), data = MDLD_speed_long, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))

anova(PL_MLM_4, PL_MLM) # p-val < 0.05

# Compute model 6: log-time, Gender * log_time, ID (intercept and slope) and Best event levels (intercept)

PL_MLM_6 <- lmer(log_speed ~ 1 + log_time + Gender * log_time + (1 + log_time | ID) + (1 | Best_event), data = MDLD_speed_long, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))

anova(PL_MLM, PL_MLM_6) # p-val < 0.05 

# pval getting lower, lower improvements and convergence issues --> PL_MLM is selected
# There is no need to go further into the complexity

#=== 2. PL MLM

# 2.1. Extracting the fixed and random effects

fixed_effects_PL_MLM <- fixef(PL_MLM)
random_effects_ID_PL_MLM <- ranef(PL_MLM)$ID
random_effects_BestEvent_PL_MLM <- ranef(PL_MLM)$Best_event

# Create a data frame to store the results

PL_MLM_res <- MDLD_speed[, c(1:3)]

# 2.2. Compute the Speed parameter (S)

# S is the exponential of the intercepts (Fixed and Random Effects)
# S = exp(gamma_{000} + b_{0j} + u_{0i} + gamma_{200} * Gender_{ij})
# where: gamma_{000} = fixed intercept
#        b_{0j} = event j random intercept
#        u_{0i} = individual-specific random deviation intercept for athlete i;
#        gamma_{200} = fixed effect representing population-level relationships for Gender; 

PL_MLM_res <- PL_MLM_res %>%
  mutate(
    
    # Extract random effects for intercept
    random_effect_event_intercept = random_effects_BestEvent_PL_MLM$`(Intercept)`[match(as.character(Best_event), rownames(random_effects_BestEvent_PL_MLM))], # b_{0j}
    random_effect_id_intercept = random_effects_ID_PL_MLM$`(Intercept)`[match(as.character(ID), rownames(random_effects_ID_PL_MLM))], # u_{0i}
    
    # Calculate Speed (S)
    S = exp(
      fixed_effects_PL_MLM["(Intercept)"] + # gamma_{000}
        random_effect_event_intercept + # b_{0j}, matched to the event
        random_effect_id_intercept + # u_{0i}, matched to the individual
        fixed_effects_PL_MLM["GenderFemale"] * as.numeric(Gender == "Female") # gamma_{200} * Gender_{ij}
    )
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_event_intercept, -random_effect_id_intercept)

# Summary of the data

summary(PL_MLM_res$S)

# Check for errors in matching effects

if(any(is.na(random_effects_ID_PL_MLM$`(Intercept)`))) {
  warning("Some IDs are not matched correctly to random effects.")
}

# 2.3. Compute the Endurance parameter (E)

# E = 1 + slope
# E = 1 + gamma_{100} + u_{1i}
# where: gamma_{100} = fixed slope
#        u_{1i} = random slope of the individual i

# Revised computation for Endurance (E)
PL_MLM_res <- PL_MLM_res %>%
  mutate(
    
    # Extract random effects for slope (log_time)
    random_effect_id_slope = random_effects_ID_PL_MLM$log_time[match(as.character(ID), rownames(random_effects_ID_PL_MLM))],
    
    # Calculate Endurance (E)
    E = 1 + fixed_effects_PL_MLM["log_time"] + # gamma_{100}
      random_effect_id_slope # u_{1i}, matched to the individual
      ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_id_slope)

# Summary of the parameter

summary(PL_MLM_res$E)

# Check for errors in matching effects

if(any(is.na(random_effects_ID_PL_MLM$log_time))) {
  warning("Some IDs are not matched correctly to random effects.")
}

# 2.4. Compute the a and b parameters and calculate the predictions

# a = The scaling coefficient is derived from the intercept terms (similar to S) but represents 
#     the raw scaling in the power-law equation:
# a = exp(Intercept + Random Effects for Intercepts + Fixed Effect Modifiers)

PL_MLM_res <- PL_MLM_res %>%
  mutate(
    
    # Extract random effects for intercept
    random_effect_event_intercept = random_effects_BestEvent_PL_MLM$`(Intercept)`[match(as.character(Best_event), rownames(random_effects_BestEvent_PL_MLM))], # b_{0j}
    random_effect_id_intercept = random_effects_ID_PL_MLM$`(Intercept)`[match(as.character(ID), rownames(random_effects_ID_PL_MLM))], # u_{0i}
    
    # Calculate scaling coefficient is derived from the intercept terms (a)
    a = exp(
      fixed_effects_PL_MLM["(Intercept)"] + 
        random_effect_event_intercept + 
        random_effect_id_intercept + 
        fixed_effects_PL_MLM["GenderFemale"] * as.numeric(Gender == "Female")
    )
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_event_intercept, -random_effect_id_intercept)

# Summary of the data

summary(PL_MLM_res$a)
summary(PL_MLM_res$S)

# b = The decay rate of the power-law model depends on the slope terms:
# b = -(slope + Random Effects for Slopes)

PL_MLM_res <- PL_MLM_res %>%
  mutate(
    
    # Extract random effects for slope (log_time)
    random_effect_id_slope = random_effects_ID_PL_MLM$log_time[match(as.character(ID), rownames(random_effects_ID_PL_MLM))],
    
    # Calculate b
    b = -(fixed_effects_PL_MLM["log_time"] + random_effect_id_slope)
    
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_id_slope)

# Summary of the parameter

summary(PL_MLM_res$b)

# 2.5. Merge results

# Merge PL_MLM_res with MDLD_speed

MDLD_speed <- left_join(MDLD_speed, PL_MLM_res[,c(1,4:7)], by="ID")

# 2.5. Save

# Save models and datasets

save(PL_MLM, file="PL_MLM")
save(PL_MLM_res, file="PL_MLM_res")

#=== 3. CS MLM

# 3.1. Compute the model follwing the same structure as PL MLM 
# Equation : distance_{ijk} = delta_{000} + p_{0j} + v_{0i} + delta_{200} * Gender_{ij} + 
#                            (delta_{100} + v_{1i}) * T_{ijk} + epsilon_{ijk}^{CS}.

CS_MLM <- lmer(distance ~ 1 + time + Gender + (1 + time | ID) + (1 | Best_event), data = MDLD_speed_long, REML = FALSE,
               control=lmerControl(optimizer = "bobyqa", check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))

# 3.2. Extracting the fixed and random effects

fixed_effects_CS_MLM <- fixef(CS_MLM)
random_effects_ID_CS_MLM <- ranef(CS_MLM)$ID
random_effects_BestEvent_CS_MLM <- ranef(CS_MLM)$Best_event

# Create a data frame to store the results

CS_MLM_res <- MDLD_speed[, c(1:3)]

# 3.3. Compute the Speed parameter (D')

# D' = intercept
# D' = delta_{000} + p_{0j} + v_{0i} + delta_{200} * Gender_{ij}
# where: delta_{000} = fixed intercept
#        p_{0j} = event j random intercept
#        v_{0i} = individual-specific random deviation intercept for athlete i;
#        delta_{200} = fixed effect representing population-level relationships for Gender; 

CS_MLM_res <- CS_MLM_res %>%
  mutate(
    
    # Extract random effects related to D'
    random_effect_event_intercept = random_effects_BestEvent_CS_MLM$`(Intercept)`[match(as.character(Best_event), rownames(random_effects_BestEvent_CS_MLM))],
    random_effect_id_intercept = random_effects_ID_CS_MLM$`(Intercept)`[match(as.character(ID), rownames(random_effects_ID_CS_MLM))],
    
    # Calculating D' with fixed and random effects
    D_prime = fixed_effects_CS_MLM["(Intercept)"] + # delta_{000}
      random_effect_event_intercept + # p_{0j}, matched correctly to the best event
      random_effect_id_intercept + # v_{0i}, matched correctly to the individual ID
      fixed_effects_CS_MLM["GenderFemale"] * as.numeric(Gender == "Female") # delta_{200} * Gender_{ij}
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_event_intercept, -random_effect_id_intercept)

# Summary of the parameter

summary(CS_MLM_res$D_prime)

# Check for errors in matching effects

if(any(is.na(random_effects_ID_CS_MLM$`(Intercept)`))) {
  warning("Some IDs are not matched correctly to random effects.")
}

# 3.4. Compute the Endurance parameter (CS)

# CS = slope * 3.6 (for a km/h output)
# CS = (delta_{100} + v_{1i}) * 3.6
# where: delta_{100} = fixed slope
#        v_{1i} = random slope of the individual i

# Revised computation for CS_ms using vectorized approach
CS_MLM_res <- CS_MLM_res %>%
  mutate(
    
    # Extract random effects related to CS
    random_effect_id_slope = random_effects_ID_CS_MLM$time[match(as.character(ID), rownames(random_effects_ID_CS_MLM))],
    
    # Calculate CS_ms (in m/s)
    CS_ms = fixed_effects_CS_MLM["time"] + # delta_{100}
      random_effect_id_slope # v_{1i}, matched to the individual
    ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_id_slope)

CS_MLM_res$CS <- CS_MLM_res$CS_ms * 3.6

# Summary of the parameter

summary(CS_MLM_res$CS_ms)
summary(CS_MLM_res$CS)

# Check for errors in matching effects

if(any(is.na(random_effects_ID_CS_MLM$time))) {
  warning("Some IDs are not matched correctly to random effects.")
}

# 3.5. Merge results

# Merge CS_MLM_res with MDLD_speed

MDLD_speed <- left_join(MDLD_speed, CS_MLM_res[,c(1,4:6)], by="ID")

# 3.6. Save

# Save models and datasets

save(CS_MLM, file="CS_MLM")
save(CS_MLM_res, file="CS_MLM_res")

#=== 4. Predictions

# 4.1. Predictions on PL MLM

# Merge E, S, a and b
MDLD_speed_long <- left_join(MDLD_speed_long, MDLD_speed[, c("ID", "S", "E", "a", "b", "D_prime", "CS_ms", "CS")], by = "ID")

# Predict speed and time using a and b
MDLD_speed_long <- MDLD_speed_long %>%
  mutate(
    pred_time_PL = (a / distance)^(1 / (b - 1)),
    speed_pred_PL = distance / pred_time_PL
  )

# 4.2. Predictions on CS MLM

# Merge CS_m, CS and D_prime

MDLD_speed_long <- left_join(MDLD_speed_long, MDLD_speed[, c(1, 26:28)], by = "ID")

# Predictions are made using the following equation: Prediction_time = (Distance - D') / CS, with Distance and D' in m and CS in m/s
MDLD_speed_long$pred_time_CS <- (as.numeric(MDLD_speed_long$distance) - MDLD_speed_long$D_prime) / MDLD_speed_long$CS_ms

# Predicted speed = distance / predicted_time
MDLD_speed_long$speed_pred_CS <- as.numeric(MDLD_speed_long$distance) / MDLD_speed_long$pred_time_CS

# 4.3. Compute and store errors

# 4.3.1. Detect outliers to choose error method

# Function to detect outliers using IQR method
detect_outliers_iqr <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[column]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  data %>% mutate(is_outlier = ifelse(data[[column]] < lower_bound | data[[column]] > upper_bound, TRUE, FALSE))
}

# Function to detect outliers using z-score method
detect_outliers_zscore <- function(data, column, threshold = 3) {
  data %>% mutate(is_outlier = ifelse(abs(scale(data[[column]])) > threshold, TRUE, FALSE))
}

# Apply outlier detection to speed and time predictions
outliers_time <- detect_outliers_iqr(MDLD_speed_long, "time")
outliers_speed <- detect_outliers_iqr(MDLD_speed_long, "speed")

# Count outliers per event
outlier_summary <- MDLD_speed_long %>%
  mutate(
    time_outlier = detect_outliers_iqr(MDLD_speed_long, "time")$is_outlier,
    speed_outlier = detect_outliers_iqr(MDLD_speed_long, "speed")$is_outlier
  ) %>%
  group_by(distance) %>%
  summarise(
    time_outliers = sum(time_outlier, na.rm = TRUE),
    speed_outliers = sum(speed_outlier, na.rm = TRUE),
    total_entries = n()
  ) %>%
  mutate(
    time_outlier_rate = round(100 * time_outliers / total_entries, 2),
    speed_outlier_rate = round(100 * speed_outliers / total_entries, 2)
  )

# Boxplots to visualize outliers
ggplot(MDLD_speed_long, aes(x = as.factor(distance), y = time)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 16) +
  labs(title = "Boxplot of Time Predictions with Outliers", x = "Distance", y = "Time") +
  theme_minimal()

ggplot(MDLD_speed_long, aes(x = as.factor(distance), y = speed)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 16) +
  labs(title = "Boxplot of Speed Predictions with Outliers", x = "Distance", y = "Speed") +
  theme_minimal()

# Print outlier summary
print(outlier_summary)

# Decision based on the number of outliers
if (mean(outlier_summary$time_outlier_rate) > 5 | mean(outlier_summary$speed_outlier_rate) > 5) {
  print("?????? High number of outliers detected. MARE/MAE is recommended for robust error computation.")
} else {
  print("??? Low number of outliers detected. RMSE/RRMSE is suitable.")
}

# 4.3.2. Compute errors 

# Function to compute RMSE, RRMSE, MAE, and MARE  for a given model (PL or CS)
compute_errors <- function(data, pred_col, true_col) {
  # Calculate overall RMSE and RRMSE
  overall_rmse <- round(sqrt(mean((data[[true_col]] - data[[pred_col]])^2, na.rm = TRUE)), 4)
  overall_rrmse <- round(overall_rmse / mean(data[[true_col]], na.rm = TRUE), 4)
  
  # Compute overall MAE
  overall_mae <- round(mean(abs(data[[true_col]] - data[[pred_col]]), na.rm = TRUE), 4)
  # Compute overall MARE (Drake's approach)
  overall_mare <- round(mean(abs(1 - data[[pred_col]] / data[[true_col]]), na.rm = TRUE), 4)
  
  # Compute errors by event
  event_errors <- data %>%
    group_by(distance) %>%
    summarise(
      MSE = mean((.data[[true_col]] - .data[[pred_col]])^2, na.rm = TRUE),
      RMSE = round(sqrt(MSE), 4),
      mean_value = mean(.data[[true_col]], na.rm = TRUE),
      RRMSE = round(RMSE / mean_value, 4),
      MAE = round(mean(abs(.data[[true_col]] - .data[[pred_col]]), na.rm = TRUE), 4),
      MARE = round(mean(abs(1 - .data[[pred_col]] / .data[[true_col]]), na.rm = TRUE), 4)
    ) %>%
    ungroup() %>%
    dplyr::select(distance, RMSE, RRMSE, MARE, MAE)
  
  return(list(overall = c(RMSE = overall_rmse, RRMSE = overall_rrmse, MARE = overall_mare, MAE = overall_mae), 
              by_event = event_errors))
}

# Function to create a structured error dataframe
create_error_df <- function(pl_errors, cs_errors, distances) {
  data.frame(
    By_event = c("All", distances),  # Only append event distances once
    PL_RMSE = c(pl_errors$overall["RMSE"], pl_errors$by_event$RMSE),
    PL_RRMSE = c(pl_errors$overall["RRMSE"], pl_errors$by_event$RRMSE),
    PL_MAE = c(pl_errors$overall["MAE"], pl_errors$by_event$MAE),
    PL_MARE = c(pl_errors$overall["MARE"], pl_errors$by_event$MARE),
    CS_RMSE = c(cs_errors$overall["RMSE"], cs_errors$by_event$RMSE),
    CS_RRMSE = c(cs_errors$overall["RRMSE"], cs_errors$by_event$RRMSE),
    CS_MAE = c(cs_errors$overall["MAE"], cs_errors$by_event$MAE),
    CS_MARE = c(cs_errors$overall["MARE"], cs_errors$by_event$MARE)
  )
}

# List of distances
event_distances <- c("400m", "800m", "1500m", "3000m", "5000m", "10,000m")

# Compute time prediction errors for PL and CS models
time_pl_errors <- compute_errors(MDLD_speed_long, "pred_time_PL", "time")
time_cs_errors <- compute_errors(MDLD_speed_long, "pred_time_CS", "time")

# Compute speed prediction errors for PL and CS models
speed_pl_errors <- compute_errors(MDLD_speed_long, "speed_pred_PL", "speed")
speed_cs_errors <- compute_errors(MDLD_speed_long, "speed_pred_CS", "speed")

# Create error dataframes for time and speed
Pred_MLM_time <- create_error_df(time_pl_errors, time_cs_errors, event_distances)
Pred_MLM_speed <- create_error_df(speed_pl_errors, speed_cs_errors, event_distances)

#=== 5. Parameters description

Parameters <- MDLD_speed %>% 
  group_by(Gender, Best_event) %>%
  summarise(
    Mean_S = round(mean(S), 2),
    sd_S = round(sd(S),2),
    Mean_E = round(mean(E), 2),
    sd_E = round(sd(E), 2),
    Mean_D_prime = round(mean(D_prime), 2),
    sd_D_prime = round(sd(D_prime), 2),
    Mean_CS = round(mean(CS), 2),
    sd_CS = round(sd(CS), 2)
  )

Parameters_level <- MDLD_speed %>% 
  group_by(Max_level, Gender, Best_event) %>%
  summarise(
    Mean_S = round(mean(S), 2),
    sd_S = round(sd(S),2),
    Mean_E = round(mean(E), 2),
    sd_E = round(sd(E), 2),
    Mean_D_prime = round(mean(D_prime), 2),
    sd_D_prime = round(sd(D_prime), 2),
    Mean_CS = round(mean(CS), 2),
    sd_CS = round(sd(CS), 2)
  )

#=== 6. Correlation between models parameters and performance in events

# How much parameters are correlated to speed in events?

# Create dataframe

event_distances <- c("400m", "800m", "1500m", "3000m", "5000m", "10000m")
param_names <- c("S", "E", "D_prime", "CS")

Perf_deter_speed <- data.frame(
  By_event = event_distances,
  matrix(ncol = length(param_names) * 2, nrow = length(event_distances))
)
colnames(Perf_deter_speed)[-1] <- c(rbind(paste0(param_names, "_corr"), paste0(param_names, "_pval")))

# Function to compute correlations with significance stars
compute_correlations <- function(param, event_speeds) {
  sapply(event_speeds, function(speed) {
    cor_test <- cor.test(MDLD_speed[[param]], MDLD_speed[[speed]])
    pval <- cor_test$p.value
    stars <- ifelse(pval < 0.001, "***", ifelse(pval < 0.01, "**", ifelse(pval < 0.05, "*", "")))
    c(corr = paste(round(cor_test$estimate, 4), stars), pval = round(pval, 4), sep = " ")
  })
}

# Compute correlations for each parameter
for (param in param_names) {
  results <- compute_correlations(param, paste0("speed_", c("400", "800", "1500", "3000", "5000", "10000")))
  Perf_deter_speed[, paste0(param, "_corr")] <- results[1, ]
  Perf_deter_speed[, paste0(param, "_pval")] <- results[2, ]
}

# Display results
print(Perf_deter_speed)

#=== 7. Real life cases 

# 7.1. Create tables

# Separate data into Male and Female

MDLD_speed_long_MAN <- filter(MDLD_speed_long, Gender == "Male")
MDLD_speed_long_WOM <- filter(MDLD_speed_long, Gender == "Female")

# Function to extract best, median, and worst performances for each event distance
get_summary_rows <- function(df) {
  df %>%
    group_by(distance) %>%
    arrange(time) %>%  # Sort each group by `time` in ascending order (best first)
    summarize(
      Best_time = round(first(time), 2),
      Best_pred_time_PL = round(first(pred_time_PL), 2),
      Best_pred_time_CS = round(first(pred_time_CS), 2),
      Best_S = round(first(S), 2),
      Best_E = round(first(E), 2),
      Best_D_prime = round(first(D_prime), 2),
      Best_CS = round(first(CS), 2),
      
      Second_time = round(nth(time, 2), 2),
      Second_pred_time_PL = round(nth(pred_time_PL, 2), 2),
      Second_pred_time_CS = round(nth(pred_time_CS, 2), 2),
      Second_S = round(nth(S, 2), 2),
      Second_E = round(nth(E, 2), 2),
      Second_D_prime = round(nth(D_prime, 2), 2),
      Second_CS = round(nth(CS, 2), 2),
      
      Third_time = round(nth(time, 3), 2),
      Third_pred_time_PL = round(nth(pred_time_PL, 3), 2),
      Third_pred_time_CS = round(nth(pred_time_CS, 3), 2),
      Third_S = round(nth(S, 3), 2),
      Third_E = round(nth(E, 3), 2),
      Third_D_prime = round(nth(D_prime, 3), 2),
      Third_CS = round(nth(CS, 3), 2),
      
      Median_time = round(median(time), 2),
      Median_pred_time_PL = round(median(pred_time_PL), 2),
      Median_pred_time_CS = round(median(pred_time_CS), 2),
      Median_S = round(median(S), 2),
      Median_E = round(median(E), 2),
      Median_D_prime = round(median(D_prime), 2),
      Median_CS = round(median(CS), 2)
    ) %>%
    pivot_longer(-distance, names_to = "Performance_Type", values_to = "Value") %>%
    separate(Performance_Type, into = c("Rank", "Variable"), sep = "_", extra = "merge")
}

# Filter male and female data
MDLD_speed_long_MAN <- filter(MDLD_speed_long, Gender == "Male")
MDLD_speed_long_WOM <- filter(MDLD_speed_long, Gender == "Female")

# Apply the function and reshape to long format
male_summary_long <- get_summary_rows(MDLD_speed_long_MAN)
female_summary_long <- get_summary_rows(MDLD_speed_long_WOM)

# Add gender column for clarity
male_summary_long <- male_summary_long %>% mutate(Gender = "Male")
female_summary_long <- female_summary_long %>% mutate(Gender = "Female")

# 7.2. Plots the profiles

# Function to extract best, median, and worst performances for each event distance
get_summary_plot <- function(df) {
  df %>%
    group_by(distance) %>%
    arrange(time) %>%  # Sort each group by `time` in ascending order (best first)
    summarize(
      Best_time = round(first(time), 2),
      Best_S = first(S),
      Best_b = first(b),
      Best_D_prime = first(D_prime),
      Best_CS = first(CS_ms),
      
      Second_time = round(nth(time, 2), 2),
      Second_S = nth(S, 2),
      Second_b = nth(b, 2),
      Second_D_prime = nth(D_prime, 2),
      Second_CS = nth(CS_ms, 2),
      
      Third_time = round(nth(time, 3), 2),
      Third_S = nth(S, 3),
      Third_b = nth(b, 3),
      Third_D_prime = nth(D_prime, 3),
      Third_CS = nth(CS_ms, 3),
      
    ) %>%
    filter(distance == 800)
}
# Apply the function and reshape to long format
male_plot_long <- get_summary_plot(MDLD_speed_long_MAN)
female_plot_long <- get_summary_plot(MDLD_speed_long_WOM)

# Power law plotting

# Define Power Law function
power_law <- function(S, b, T) {
  return(S * T^(-b))
}

# Define time range (10s to 600s)
T <- seq(10, 200)

# Given parameters for each runner
S_MAN_A <- male_plot_long$Best_S
b_MAN_A <- male_plot_long$Best_b
S_MAN_B <- male_plot_long$Second_S
b_MAN_B <- male_plot_long$Second_b
S_MAN_C <- male_plot_long$Third_S
b_MAN_C <- male_plot_long$Third_b

# Compute speed values
V_MAN_A <- power_law(S_MAN_A, b_MAN_A, T)
V_MAN_B <- power_law(S_MAN_B, b_MAN_B, T)
V_MAN_C <- power_law(S_MAN_C, b_MAN_C, T)

# Create data frame for ggplot
df_MAN <- data.frame(
  Time = rep(T, 3),
  Speed = c(V_MAN_A, V_MAN_B, V_MAN_C),
  Runner = rep(c("Male athlete A", "Male athlete B", "Male athlete C"), each = length(T))
)

# Plot the results
plot_PL_MAN_800 <- ggplot(df_MAN, aes(x = Time, y = Speed, color = Runner)) +
  geom_line(size = 1) +
  labs(title = "Power Law Relationship: Speed vs. Time",
       subtitle = "Three fastest 800 m males athletes speed-duration profiles",
       x = "Time (s)", y = "Speed (m/s)") +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 15),  # Set font
    legend.title = element_blank()
  )

# Given parameters for each runner
S_WOM_A <- female_plot_long$Best_S
b_WOM_A <- female_plot_long$Best_b
S_WOM_B <- female_plot_long$Second_S
b_WOM_B <- female_plot_long$Second_b
S_WOM_C <- female_plot_long$Third_S
b_WOM_C <- female_plot_long$Third_b

# Compute speed values
V_WOM_A <- power_law(S_WOM_A, b_WOM_A, T)
V_WOM_B <- power_law(S_WOM_B, b_WOM_B, T)
V_WOM_C <- power_law(S_WOM_C, b_WOM_C, T)

# Create data frame for ggplot
df_WOM <- data.frame(
  Time = rep(T, 3),
  Speed = c(V_WOM_A, V_WOM_B, V_WOM_C),
  Runner = rep(c("Female athlete A", "Female athlete B", "Female athlete C"), each = length(T))
)

# Plot the results
plot_PL_WOM_800 <- ggplot(df_WOM, aes(x = Time, y = Speed, color = Runner)) +
  geom_line(size = 1) +
  labs(title = "Power Law Relationship: Speed vs. Time",
       subtitle = "Three fastest 800 m females athletes speed-duration profiles",
       x = "Time (s)", y = "Speed (m/s)") +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 15),  # Set font
    legend.title = element_blank()
  ) + 
  scale_color_brewer(palette="Dark2")

# Critical speed plotting 

# Define the Critical Speed speed-duration function
critical_speed_velocity <- function(CS, D_prime, T) {
  return(CS + (D_prime / T))
}

# Given parameters for each runner
D_prime_MAN_A <- male_plot_long$Best_D_prime
CS_MAN_A <- male_plot_long$Best_CS
D_prime_MAN_B <- male_plot_long$Second_D_prime
CS_MAN_B <- male_plot_long$Second_CS
D_prime_MAN_C <- male_plot_long$Third_D_prime
CS_MAN_C <- male_plot_long$Third_CS

# Compute speed values
S_MAN_A <- critical_speed_velocity(CS_MAN_A, D_prime_MAN_A, T)
S_MAN_B <- critical_speed_velocity(CS_MAN_B, D_prime_MAN_B, T)
S_MAN_C <- critical_speed_velocity(CS_MAN_C, D_prime_MAN_C, T)

# Create data frame for ggplot
df_speed <- data.frame(
  Time = rep(T, 3),
  Speed = c(S_MAN_A, S_MAN_B, S_MAN_C),
  Runner = rep(c("Male athlete A", "Male athlete B", "Male athlete C"), each = length(T))
)

# Plot the results
plot_CS_MAN_800 <- ggplot(df_speed, aes(x = Time, y = Speed, color = Runner)) +
  geom_line(size = 1) +
  labs(title = "Critical Speed Relationship: Speed vs. Time",
       subtitle = "Three fastest 800 m males athletes speed-duration profiles",
       x = "Time (s)", y = "Speed (m/s)") +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 15),
    legend.title = element_blank()
  )

# Given parameters for each runner
D_prime_WOM_A <- female_plot_long$Best_D_prime
CS_WOM_A <- female_plot_long$Best_CS
D_prime_WOM_B <- female_plot_long$Second_D_prime
CS_WOM_B <- female_plot_long$Second_CS
D_prime_WOM_C <- female_plot_long$Third_D_prime
CS_WOM_C <- female_plot_long$Third_CS

# Compute speed values
S_WOM_A <- critical_speed_velocity(CS_WOM_A, D_prime_WOM_A, T)
S_WOM_B <- critical_speed_velocity(CS_WOM_B, D_prime_WOM_B, T)
S_WOM_C <- critical_speed_velocity(CS_WOM_C, D_prime_WOM_C, T)

# Create data frame for ggplot
df_speed <- data.frame(
  Time = rep(T, 3),
  Speed = c(S_WOM_A, S_WOM_B, S_WOM_C),
  Runner = rep(c("Female athlete A", "Female athlete B", "Female athlete C"), each = length(T))
)

# Plot the results
plot_CS_WOM_800 <- ggplot(df_speed, aes(x = Time, y = Speed, color = Runner)) +
  geom_line(size = 1) +
  labs(title = "Critical Speed Relationship: Speed vs. Time",
       subtitle = "Three fastest 800 m females athletes speed-duration profiles",
       x = "Time (s)", y = "Speed (m/s)") +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 15),  # Set font
    legend.title = element_blank()
  ) + 
  scale_color_brewer(palette="Dark2")

# Arrange plots

# Create the layout for four plots
combined_plot <- (
  (plot_PL_MAN_800 + plot_CS_MAN_800) / 
    (plot_PL_WOM_800 + plot_CS_WOM_800)
) + 
  plot_annotation(
    tag_levels = "A",  # Automatically assigns "A", "B", "C", "D"
    theme = theme(text = element_text(family = "Times New Roman"))
  ) & 
  theme(legend.position = "bottom")

# Display the final plot
print(combined_plot)