# Exploratory Data analysis
# by Kenneth Kamogelo Baloyi
# 08 July 2025

#Set working directory
setwd("C:/Users/Kenneth Kamogelo/OneDrive - University of Cape Town/Desktop/CSIR/Vac work/Code") 

# Load csv into a dataframe
df <- read.csv("Test_balanced_discharge_data.csv")

#Train the linear model
model <- lm(capacity~cycle+current_measured)
summary(model)# Basic summary
summary(df)

# Check for missing values
colSums(is.na(df))

# Check each battery how many data points it has
table(df$battery)

library(corrplot) #Load the library

numeric_vars <- df[, sapply(df, is.numeric)] #retain only numeric variables
numeric_vars <- subset(numeric_vars, select = -ambient_temperature) #subset 
#without
#ambient temperature as it is always constant

# Compute correlation matrix
cor_mat <- cor(numeric_vars)

# Plot with correlation coefficients
corrplot(cor_mat,
         method = "color",        # color shading of boxes
         type = "upper",          # show upper triangle only
         addCoef.col = "black",   # add correlation coefficients in black
         number.cex = 0.7,        # size of the numbers
         tl.cex = 0.8,            # size of the axis labels
         tl.col = "black",        # label color
         diag = FALSE) 



library(GGally)
ggpairs(numeric_vars[, c("capacity", "voltage_measured", "current_measured", 
                         "temperature_measured")]) #headers
attach(df)
library(ggplot2)
#Plot Cycle vs. capacity
ggplot(df, aes(x = cycle, y = capacity, color = battery)) +
  geom_line() +
  labs(title = "Battery Capacity over increasing Cycles", x = "Cycle count",
       y = "Capacity (Ah)")


#Plot time vs. capacity
ggplot(df, aes(x = time_cumulative, y = capacity, color = battery)) +
  geom_line() +
  labs(title = "Battery Capacity over time", x = "Time (s)",
       y = "Capacity (Ah)")

#Plot voltage measured vs. capacity
ggplot(df, aes(x = voltage_measured, y = capacity, color = battery)) +
  geom_line() +
  labs(title = "Battery Capacity with increasing voltage measured", x = "Voltage measured (V)",
       y = "Capacity (Ah)")

#Principal component analysis
# OPTIONAL
features <- c("voltage_measured", "current_measured",
              "temperature_measured", "current", "voltage", "time_cumulative")

df_scaled <- scale(df[, features])
pca_result <- prcomp(df_scaled, center = TRUE, scale. = TRUE)
# Scree plot
plot(pca_result, type = "l", main = "Scree Plot")

# Biplot
biplot(pca_result, scale = 0)
loadings <- pca_result$rotation
print(loadings)

#Analysis of Variance
features <- c("capacity", "temperature_measured", 
              "voltage_measured")

for (var in features) {
  cat("\nANOVA for", var, "\n")
  model <- aov(df[[var]] ~ battery, data = df)
  print(summary(model))
}

#post-hoc test

TukeyHSD(aov(capacity ~ battery, data = df))


summary(pca_result) #Summary of PCA

library(ggplot2)
#Boxplots
ggplot(df, aes(x = battery, y = capacity)) + 
  geom_boxplot() + 
  labs(title = "Capacity distribution across different batteries", x="Battery", y="Capacity (Ah)")
