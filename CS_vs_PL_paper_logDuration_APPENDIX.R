#######################################################################################
#                                                                                     #      
#           Using MLM to compare Power Law vs Critical speed: APPENDIX                #
#           ----------------------------------------------------------                #  
#                                                                                     #
#   0. Load mandatory material and prepare datasets                                   #
#   A. Compute the CS model without 10,000m performances                              #
#   B. Compute the PL model without 10,000m performances                              #
#   C. Compute predictions and errors and compare to all-event models                 #
#                                                                                     #
#                                                       June 2024 - February 2025     #
#                                                                                     #
#######################################################################################

# READ ME: In this script we aim to analyse the differences with and without 10,000m performances
#          in the model computation. Answering this question: 
#          How PL and CS evolve when computing with different distances?

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

Appendix <- MDLD_speed

# 0.3. Adjust the dataset to the needs of the study

# Transform the dataset to long format
Appendix_long <- Appendix %>%
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
Appendix_long <- Appendix_long %>%
  mutate(
    log_speed = log(speed),
    log_time = log(time)
  )

# Delete 10,000m performances to compute the models

Appendix_long_2 <- subset(Appendix_long, Appendix_long$distance!=10000) 

#=== A. Compute the CS model without 10,000m performances

# A.1. Compute the model follwoing the same structure and without 10,000m
# Equation : distance_{ijk} = delta_{000} + p_{0j} + v_{0i} + delta_{200} * Gender_{ij} + 
#                            (delta_{100} + v_{1i}) * T_{ijk} + epsilon_{ijk}^{CS}.

CS_MLM_appendix <- lmer(distance ~ 1 + time + Gender + (1 + time | ID) + (1 | Best_event), data = Appendix_long_2, REML = FALSE,
                        control=lmerControl(optimizer = "bobyqa", check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))

# A.2. Extracting the fixed and random effects

fixed_effects_CS_MLM_appendix <- fixef(CS_MLM_appendix)
random_effects_ID_CS_MLM_appendix <- ranef(CS_MLM_appendix)$ID
random_effects_BestEvent_CS_MLM_appendix <- ranef(CS_MLM_appendix)$Best_event

# Create a data frame to store the results

CS_MLM_appendix_res <- MDLD_speed[, c(1:3)]

# A.3. Compute the Speed parameter (D')

# D' = intercept
# D' = delta_{000} + p_{0j} + v_{0i} + delta_{200} * Gender_{ij}
# where: delta_{000} = fixed intercept
#        p_{0j} = event j random intercept
#        v_{0i} = individual-specific random deviation intercept for athlete i;
#        delta_{200} = fixed effect representing population-level relationships for Gender; 

CS_MLM_appendix_res <- CS_MLM_appendix_res %>%
  mutate(
    
    # Extract random effects related to D'
    random_effect_event_intercept = random_effects_BestEvent_CS_MLM_appendix$`(Intercept)`[match(as.character(Best_event), rownames(random_effects_BestEvent_CS_MLM_appendix))],
    random_effect_id_intercept = random_effects_ID_CS_MLM_appendix$`(Intercept)`[match(as.character(ID), rownames(random_effects_ID_CS_MLM_appendix))],
    
    # Calculating D' with fixed and random effects
    D_prime = fixed_effects_CS_MLM_appendix["(Intercept)"] + # delta_{000}
      random_effect_event_intercept + # p_{0j}, matched correctly to the best event
      random_effect_id_intercept + # v_{0i}, matched correctly to the individual ID
      fixed_effects_CS_MLM_appendix["GenderFemale"] * as.numeric(Gender == "Female") # delta_{200} * Gender_{ij}
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_event_intercept, -random_effect_id_intercept)

# Summary of the parameter

summary(CS_MLM_appendix_res$D_prime)

# Check for errors in matching effects

if(any(is.na(random_effects_ID_CS_MLM_appendix$`(Intercept)`))) {
  warning("Some IDs are not matched correctly to random effects.")
}

# A.4. Compute the Endurance parameter (CS)

# CS = slope * 3.6 (for a km/h output)
# CS = (delta_{100} + v_{1i}) * 3.6
# where: delta_{100} = fixed slope
#        v_{1i} = random slope of the individual i

# Revised computation for CS_ms using vectorized approach
CS_MLM_appendix_res <- CS_MLM_appendix_res %>%
  mutate(
    
    # Extract random effects related to CS
    random_effect_id_slope = random_effects_ID_CS_MLM_appendix$time[match(as.character(ID), rownames(random_effects_ID_CS_MLM_appendix))],
    
    # Calculate CS_ms (in m/s)
    CS_ms = fixed_effects_CS_MLM_appendix["time"] + # delta_{100}
      random_effect_id_slope # v_{1i}, matched to the individual
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_id_slope)

CS_MLM_appendix_res$CS <- CS_MLM_appendix_res$CS_ms * 3.6

# Summary of the parameter

summary(CS_MLM_appendix_res$CS_ms)
summary(CS_MLM_appendix_res$CS)

# Check for errors in matching effects

if(any(is.na(random_effects_ID_CS_MLM_appendix$time))) {
  warning("Some IDs are not matched correctly to random effects.")
}

# A.5. Merge results

# Merge CS_MLM_appendix_res with MDLD_speed

Appendix <- left_join(Appendix, CS_MLM_appendix_res[,c(1,4:6)], by="ID")

#=== B. Compute the PL model without 10,000m performances 

# B.1. Compute the model follwoing the same structure and without 10,000m

PL_MLM_appendix <- lmer(log_speed ~ 1 + log_time + Gender + (1 + log_time | ID) + (1 | Best_event), data = Appendix_long_2, REML = FALSE,
                        control = lmerControl(optimizer = "bobyqa"))

# B.2. Extracting the fixed and random effects

fixed_effects_PL_MLM_appendix <- fixef(PL_MLM_appendix)
random_effects_ID_PL_MLM_appendix <- ranef(PL_MLM_appendix)$ID
random_effects_BestEvent_PL_MLM_appendix <- ranef(PL_MLM_appendix)$Best_event

# Create a data frame to store the results

PL_MLM_appendix_res <- MDLD_speed[, c(1:3)]

# B.3. Compute the Speed parameter (S)

# S is the exponential of the intercepts (Fixed and Random Effects)
# S = exp(gamma_{000} + b_{0j} + u_{0i} + gamma_{200} * Gender_{ij})
# where: gamma_{000} = fixed intercept
#        b_{0j} = event j random intercept
#        u_{0i} = individual-specific random deviation intercept for athlete i;
#        gamma_{200} = fixed effect representing population-level relationships for Gender; 

PL_MLM_appendix_res <- PL_MLM_appendix_res %>%
  mutate(
    
    # Extract random effects for intercept
    random_effect_event_intercept = random_effects_BestEvent_PL_MLM_appendix$`(Intercept)`[match(as.character(Best_event), rownames(random_effects_BestEvent_PL_MLM_appendix))], # b_{0j}
    random_effect_id_intercept = random_effects_ID_PL_MLM_appendix$`(Intercept)`[match(as.character(ID), rownames(random_effects_ID_PL_MLM_appendix))], # u_{0i}
    
    # Calculate Speed (S)
    S = exp(
      fixed_effects_PL_MLM_appendix["(Intercept)"] + # gamma_{000}
        random_effect_event_intercept + # b_{0j}, matched to the event
        random_effect_id_intercept + # u_{0i}, matched to the individual
        fixed_effects_PL_MLM_appendix["GenderFemale"] * as.numeric(Gender == "Female") # gamma_{200} * Gender_{ij}
    )
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_event_intercept, -random_effect_id_intercept)

# Summary of the data

summary(PL_MLM_appendix_res$S)

# Check for errors in matching effects

if(any(is.na(random_effects_ID_PL_MLM_appendix$`(Intercept)`))) {
  warning("Some IDs are not matched correctly to random effects.")
}

# B.4. Compute the Endurance parameter (E)

# E = 1 + slope
# E = 1 + gamma_{100} + u_{1i}
# where: gamma_{100} = fixed slope
#        u_{1i} = random slope of the individual i

# Revised computation for Endurance (E)
PL_MLM_appendix_res <- PL_MLM_appendix_res %>%
  mutate(
    
    # Extract random effects for slope (log_time)
    random_effect_id_slope = random_effects_ID_PL_MLM_appendix$log_time[match(as.character(ID), rownames(random_effects_ID_PL_MLM_appendix))],
    
    # Calculate Endurance (E)
    E = 1 + fixed_effects_PL_MLM_appendix["log_time"] + # gamma_{100}
      random_effect_id_slope # u_{1i}, matched to the individual
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_id_slope)

# Summary of the parameter

summary(PL_MLM_appendix_res$E)

# Check for errors in matching effects

if(any(is.na(random_effects_ID_PL_MLM_appendix$log_time))) {
  warning("Some IDs are not matched correctly to random effects.")
}

# B.5. Compute the a and b parameters and calculate the predictions

# a = The scaling coefficient is derived from the intercept terms (similar to S) but represents 
#     the raw scaling in the power-law equation:
# a = exp(Intercept + Random Effects for Intercepts + Fixed Effect Modifiers)

PL_MLM_appendix_res <- PL_MLM_appendix_res %>%
  mutate(
    
    # Extract random effects for intercept
    random_effect_event_intercept = random_effects_BestEvent_PL_MLM_appendix$`(Intercept)`[match(as.character(Best_event), rownames(random_effects_BestEvent_PL_MLM_appendix))], # b_{0j}
    random_effect_id_intercept = random_effects_ID_PL_MLM_appendix$`(Intercept)`[match(as.character(ID), rownames(random_effects_ID_PL_MLM_appendix))], # u_{0i}
    
    # Calculate scaling coefficient is derived from the intercept terms (a)
    a = exp(
      fixed_effects_PL_MLM_appendix["(Intercept)"] + 
        random_effect_event_intercept + 
        random_effect_id_intercept + 
        fixed_effects_PL_MLM_appendix["GenderFemale"] * as.numeric(Gender == "Female")
    )
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_event_intercept, -random_effect_id_intercept)

# Summary of the data

summary(PL_MLM_appendix_res$a)
summary(PL_MLM_appendix_res$S)

# b = The decay rate of the power-law model depends on the slope terms:
# b = -(slope + Random Effects for Slopes)

PL_MLM_appendix_res <- PL_MLM_appendix_res %>%
  mutate(
    
    # Extract random effects for slope (log_time)
    random_effect_id_slope = random_effects_ID_PL_MLM_appendix$log_time[match(as.character(ID), rownames(random_effects_ID_PL_MLM_appendix))],
    
    # Calculate b
    b = -(fixed_effects_PL_MLM_appendix["log_time"] + random_effect_id_slope)
    
  ) %>%
  # Drop intermediate columns
  dplyr::select(-random_effect_id_slope)

# Summary of the parameter

summary(PL_MLM_appendix_res$b)

# B.6. Merge results

# Merge PL_MLM_appendix_res with MDLD_speed

Appendix <- left_join(Appendix, PL_MLM_appendix_res[,c(1,4:7)], by="ID")

#=== C. Compute predictions and errors and compare to all-event models

# C.1. Compute the predictions

# Merge CS, CS_m, D', E, S, a and b
Appendix_long <- left_join(Appendix_long, Appendix[, c("ID", "S", "E", "a", "b", "D_prime", "CS_ms", "CS")], by = "ID")

# Predict speed and time using a and b
Appendix_long <- Appendix_long %>%
  mutate(
    pred_time_PL = (a / distance)^(1 / (b - 1)),
    speed_pred_PL = distance / pred_time_PL
  )

# Predictions are made using the following equation: Prediction_time = (Distance - D') / CS, with Distance and D' in m and CS in m/s
Appendix_long$pred_time_CS <- (as.numeric(Appendix_long$distance) - Appendix_long$D_prime) / Appendix_long$CS_ms

# Predicted speed = distance / predicted_time
Appendix_long$speed_pred_CS <- as.numeric(Appendix_long$distance) / Appendix_long$pred_time_CS

# C.2. Compute the errors

# Compute time prediction errors for PL and CS models
time_pl_appendix_errors <- compute_errors(Appendix_long, "pred_time_PL", "time")
time_cs_appendix_errors <- compute_errors(Appendix_long, "pred_time_CS", "time")

# Compute speed prediction errors for PL and CS models
speed_pl_appendix_errors <- compute_errors(Appendix_long, "speed_pred_PL", "speed")
speed_cs_appendix_errors <- compute_errors(Appendix_long, "speed_pred_CS", "speed")

# Create error dataframes for time and speed
Pred_MLM_app_time <- create_error_df(time_pl_appendix_errors, time_cs_appendix_errors, event_distances)
Pred_MLM_app_speed <- create_error_df(speed_pl_appendix_errors, speed_cs_appendix_errors, event_distances)

# C.3. How much parameters are correlated to speed in events?

# Create dataframe

event_distances <- c("400m", "800m", "1500m", "3000m", "5000m", "10000m")
param_names <- c("S", "E", "D_prime", "CS")

Perf_deter_speed_app <- data.frame(
  By_event = event_distances,
  matrix(ncol = length(param_names) * 2, nrow = length(event_distances))
)
colnames(Perf_deter_speed_app)[-1] <- c(rbind(paste0(param_names, "_corr"), paste0(param_names, "_pval")))

# Function to compute correlations with significance stars
compute_correlations <- function(param, event_speeds) {
  sapply(event_speeds, function(speed) {
    cor_test <- cor.test(Appendix[[param]], Appendix[[speed]])
    pval <- cor_test$p.value
    stars <- ifelse(pval < 0.001, "***", ifelse(pval < 0.01, "**", ifelse(pval < 0.05, "*", "")))
    c(corr = paste(round(cor_test$estimate, 4), stars), pval = round(pval, 4), sep = " ")
  })
}

# Compute correlations for each parameter
for (param in param_names) {
  results <- compute_correlations(param, paste0("speed_", c("400", "800", "1500", "3000", "5000", "10000")))
  Perf_deter_speed_app[, paste0(param, "_corr")] <- results[1, ]
  Perf_deter_speed_app[, paste0(param, "_pval")] <- results[2, ]
}

# Display results
print(Perf_deter_speed_app)