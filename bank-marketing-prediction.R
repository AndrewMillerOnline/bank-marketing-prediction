library(rpart)
library(rpart.plot)
library(ggplot2)
library(caret)
library(randomForest)
library(ROSE)
library(vioplot)
library(ggpubr)

df_full <- read.csv('bank-additional-full.csv', header = TRUE, sep=";")

###########################
#### Step 1: Data Prep ####
###########################

str(df_full)

# Copy output variable into new working copy of data frame
df_new <- cbind(df_full)

# Convert job, marital, education, contact, month, day of week to factor
df_new$job <- as.factor(df_full$job)
df_new$marital <- as.factor(df_full$marital)
df_new$education <- as.factor(df_full$education)
df_new$contact <- as.factor(df_full$contact)
df_new$day_of_week <- as.factor(df_full$day_of_week)
df_new$poutcome <- as.factor(df_full$poutcome)

# Convert campaign and previous to numeric
df_new$campaign <- as.numeric(df_full$campaign)
df_new$previous <- as.numeric(df_full$previous)

# Convert default, housing, and loan to boolean, set missing rows to NA
df_new$default <- ifelse(df_full$default == "yes", 1,
                         ifelse(df_full$default == "no", 0, NA))

df_new$housing <- ifelse(df_full$housing == "yes", 1,
                         ifelse(df_full$housing == "no", 0, NA))

df_new$loan <- ifelse(df_full$loan == "yes", 1,
                      ifelse(df_full$loan == "no", 0, NA))

# Remove the duration column
df_new <- subset(df_new, select = -duration)

# Helper function to calculate the mode for a given vector
imputeMode <-function(inputData){
  uniqueData <-unique (inputData)
  uniqueData [which.max(tabulate(match(inputData,uniqueData)))]
}

# Reorder months to be chronological
df_full$month <- as.factor(df_full$month)
levels(df_full$month)
df_new$month <- factor(df_full$month, 
                  levels = c("mar", "apr", "may", "jun", "jul", 
                             "aug", "sep", "oct", "nov", "dec"))

# Convert the outcome variable to leveled factor
df_new$y <- factor(df_new$y, levels = c("yes", "no"))

# Check null values
sum(is.na(df_new$default))
sum(is.na(df_new$housing))
sum(is.na(df_new$loan))

# Impute the missing values for housing and loan
df_new$housing[is.na(df_new$housing)] <- imputeMode(df_new$housing)
df_new$loan[is.na(df_new$loan)] <- imputeMode(df_new$loan)

# Too many nulls in default, drop it
df_new <- subset(df_new, select = -default)

############################
##  STEP 2: Data Analysis ##
############################


#age: let's use a violin plot to analyze the age distribution, similar to a population pyramid
vioplot(df_new$age,
        col = 5)

#job: Retirees and students seem more likely to open an account
ggplot(df_new, aes(x=job, fill=y)) + 
  geom_bar(position="fill") + 
  theme(axis.text.x = element_text(angle=45))

#marital status
ggplot(df_new,aes(x=marital, fill=y))+geom_bar(position="fill")

#education: Illiterate is very infrequent in education, but converts more often
barplot(table(df_new$education))
ggplot(df_new,aes(x=education, fill=y)) + 
  geom_bar(position="fill") +
  theme(axis.text.x = element_text(angle=45))+ 
  scale_y_continuous(labels = scales::percent) +
  ylab("Sample Percent") + 
  labs(fill = "Convert")

#contact: cellular sees more conversions
ggplot(df_new,aes(x=contact, fill=y))+geom_bar(position="fill")

# No discernible difference based on day_of_week, so we'll omit this variable from the model
ggplot(df_new,aes(x=day_of_week, fill=y))+geom_bar(position="fill")
df_new <- subset(df_new, select = -day_of_week)

# Previous customers are much more likely to subscribe again!
ggplot(df_new,aes(x=poutcome, fill=factor(y)))+geom_bar(position="fill")


# Certain months have much higher sign-ups
## March, Sep, Oct, and Dec are highest
ggplot(df_new,aes(x=month, fill=y))+geom_bar(position="fill")

# Let's analyze the macroeconomic variables in more depth
hist(df_new$emp.var.rate)
hist(df_new$cons.price.idx)
hist(df_new$cons.conf.idx)
hist(df_new$euribor3m)
hist(df_new$nr.employed)

## Conversions are highest in the months with lowest euribor3m: 
## This will likely be an important feature
ggplot(df_new, aes(x = month, y = euribor3m)) + geom_boxplot()

# nr.employed seems to share similar trends as the euribor rate
ggplot(df_new, aes(x = month, y = nr.employed)) + geom_boxplot()

# Let's look deeper into the relationship between the macroeconomic features
g1 <- ggscatter(df_new,x = "euribor3m", y = "emp.var.rate", color="#009fe8", 
                size=1.5, shape=1) +
                border() + stat_cor()
g2 <- ggscatter(df_new,x = "euribor3m", y = "nr.employed", color="#cc87ea", 
                size=1.5, shape=1) +
                border() + stat_cor()
g3 <- ggscatter(df_new,x = "euribor3m", y = "cons.conf.idx", color="#ff708f", 
                size=1.5, shape=1) +
                border() + stat_cor()
g4 <- ggscatter(df_new,x = "euribor3m", y = "cons.price.idx", color="#ffa600", 
                size=1.5, shape=1) +
                border() + stat_cor()

figure <- ggarrange(g1, g2, g3, g4, ncol = 2, nrow = 2)
annotate_figure(figure, 
                top = text_grob("Macroeconomic Variables", color="black",
                face="bold", size=14))


#' nr.employed and emp.var.rate are both VERY highly correlated to the euribor rate
#' My intuition is that interest rates are likely the more important feature
#' so we'll drop nr.employed and emp.var.rate from the model
df_new <- subset(df_new, select = -nr.employed)
df_new <- subset(df_new, select = -emp.var.rate)

#' while the cons.conf.idx and cons.price idx look very similar on the chart above
#' they're not actually too correlated to each other.  We'll keep both features.
ggscatter(df_new,x = "cons.conf.idx", y = "cons.price.idx", color="#ffa600", 
          size=1.5, shape=1) +
  border() + stat_cor()


###############################
### Step 3: Build the model ###
###############################

# Create training and test sets
set.seed(42)
train <- sample(nrow(df_new), nrow(df_new)*.8)
training_data <- df_new[train,]
test_data <- df_new[-train,]

table(training_data$y)

# Use oversampling to build a dataset with better positive response representation
train.rose <- ROSE(y ~ .,
                   data = training_data,
                   p=.77,
                   N=16000,
                   seed = 42)$data

levels(train.rose$y)
train.rose$y <- relevel(train.rose$y, "yes")
table(train.rose$y)


# Create the model

rf <- randomForest(formula = train.rose$y~.,
                  data = train.rose,
                  ntree=1000,
                  replace = FALSE)



##################################
### Step 4: Evaluate the model ###
##################################

pred.train <- predict(rf, training_data)
confusionMatrix(pred.train, training_data$y)


predictions <- predict(rf, test_data)
confusionMatrix(predictions, test_data$y)


# Plot the lift chart
predictions.pct <- predict(rf, test_data, type = "prob")
df_lift <- cbind(test_data)
df_lift$actual <- ifelse(test_data$y=="yes", 1, 0)
df_lift$prob <- as.data.frame(predictions.pct)$yes


lift.ex <- lift(relevel(as.factor(actual), ref = "1") ~ prob, data = df_lift)
xyplot(lift.ex, plot = "gain")