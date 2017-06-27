set.seed(123)
require(caret)
library(plyr)
library(caTools)
library(remotes)
library(woe)
library(ggplot2)
library(aod)
library(ResourceSelection)
library(ROCR)
library(reshape2)
library(reshape)
library(SDMTools)
library(dplyr)

# Load the train and test set
train <- read.csv(file="CAX_Startup_Train.csv", header=TRUE,as.is=T) 
test <- read.csv(file="CAX_Startup_Test.csv", header=TRUE,as.is=T)
# Add a "Category" column
train$Category <- c("Train")
test$Category <- c("Test")
# rbind the test and train set in a bi
d <- rbind(train, test)

sapply(d, class)

# Recoding ordinal variables
d$employee_count_code <- as.numeric(revalue(d$Founders_previous_company_employee_count,
                                            c("Small"=1, "Medium"=2, "Large"=3)))
   # This variable is already eliminated from the char_data dataframe
# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 
sapply(d, function(x){sum(is.na(x))})

# Check which variables are numeric and make the cnt_data
numeric_var <- sapply(d, is.numeric)
numeric_cols <- which(numeric_var == 1)
cnt_data <- d[, numeric_cols]
# Now make the char_data 
char_data <- d[, !(colnames(d) %in% colnames(cnt_data))]

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 
d$Founders_previous_company_employee_count <- NULL

# ||| Making your own test data |||
# Separating 0 and the 1 level of the Dependent Variable
train_0 <- train[train$Dependent == 0, ]
train_1 <- train[train$Dependent == 1, ]
train_0$Category <- NULL
train_1$Category <- NULL

# randomly choosing test and train set for each level
sample_0 <- sample.split(train_0, SplitRatio = 0.9)
train_0_new <- subset(train_0, sample_0 == TRUE)
test_0_new <- subset(train_0, sample_0 == FALSE)

sample_1 = sample.split(train_1, SplitRatio = 0.9) # This is a vector of T(90%) and F(10%) of the train_1 dataset (todas las columnas pero que la variable dependiente es 1) 
train_1_new <- subset(train_1, sample_1 == TRUE) 
test_1_new <- subset(train_1, sample_1 == FALSE)
# Hasta aquí lo que primero se hizo fue separar el "train" set original en 2, los que tienen la 
# variable independiente igual a 1 en train_1 y los que tienen la variable independiente igual a 0
# en train_0
# De ahí lo que se hizo fue el obtener T's y F's en sample_0 y en sample_1
# train_0_new es el dataset de entrenamiento que tiene la variable dependiente igual a 0
# test_0_new es el dataset de prueba que tiene la variable dependiente igual a 0
# train_1_new es el dataset de entrenamiento que tiene la variable dependiente igual a 1
# test_1_new es el dataset de prueba que tiene la variable depdendiente igaul a 1

# Final new train and test set
train_new <- rbind(train_1_new, train_0_new)
test_new <- rbind(test_1_new, test_0_new)


# ||| Variable Selection Using Information Value |||
# Only dependent and independent variable of training set should be there in dataframe
train_new$CAX_ID <- NULL

filas <- row.names(train_new) # Para no olvidar las posiciones de las filas que se quieren  

# Calculation of information value
# Aparently the row names are not in order so first assign ordered numbers
row.names(train_new) <- 1:nrow(train_new)
IV <- iv.mult(train_new[ , -1], y="Dependent", TRUE)
  # Ignore warning message for variable for which WOE
IV

# Selecting variables with 0.1<IV<0.5
var <- IV[which(IV$InformationValue > 0.1), ]
var1 <- var[which(var$InformationValue < 0.5), ]
final_var <- var1$Variable

# x_train is the dataset that has all the important variables and CAX_ID
x_train <- train_new[final_var]   # This command worked
    # Lets try with the row names: filas    This seems to be the same, so train_final will be the set for modeling
set_de_prueba <- train[filas,]
Dependent <- set_de_prueba$Dependent
train_final <- cbind(Dependent, x_train)



# 5.- ||| Model Building |||
# Fitting stepwise binary logistic regression with logit link function
mod <- step(glm(Dependent~., family = binomial(link = logit), data = train_final))
summary(mod)

# Giving the results of the model we will test other model by just including some variables
model <- glm(formula = Dependent~Company_repeat_investors_count + Founders_Data_Science_skills_score + 
               Company_1st_investment_time + Founders_Domain_skills_score + Company_top_Angel_VC_funding, 
               family = binomial(link=logit), data = train_final)

# It appears that the best model is the first one:  "mod"
# 
exp(cbind(OR = coef(mod), confint(mod)))
confint(mod)
coef(mod)
# AIC - Test if the model is better than just with an intercept
with(mod, null.deviance - deviance) # 45.1191
  # The degrees of freedom for the difference between the two models is equal to 
  # the number of predictor variables in the model, 
with(mod, df.null - df.residual) # 9
  # Finally we obtain the p-value
with(mod, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE)) # 8.77e-07

    # So the chi-squared of 45.1191 with 9 degrees of freedom and an associated 
    # p-value of 8.77e-07 tells us that our model as a whole fits significantly better
    # than an empty model. This is sometimes called a likelyhood ratio test
    # (the deviance residual is -2*log likelihood). To see the models log likelihood:

    logLik(mod)  # -120.2191 (df = 10)

# To test the overall goodness of fit of your logistic regression model 
# the 'Hosmer-Lemeshow' test can be done
hoslem.test(train_new$Dependent, mod$fitted.values, g=10)
    # the p-value is 0.5701 indicating there is no evidence of poor fit, the model is 
    # fitting the data well



# 6.- ||| Predicting Test Score and Model Evaluation |||

# Prediction on test set
pred_prob <- predict(mod, newdata = test_new, type="response")

# Model accuracy measures
pred <- prediction(pred_prob, test_new$Dependent) # ROCR
# Area under the curve
performance(pred, "auc")  # ROCR
# Creating ROC curve
roc <- performance(pred, "tpr", "fpr")
plot(roc)

# The ROC Curve can be seen for probability point causing largest separation 
# between ‘TPR’ and ‘FPR’. For this we can plot the values as given below:
# Create data frame of values
perf <- as.data.frame(cbind(roc@alpha.values[[1]], roc@x.values[[1]], roc@y.values[[1]]))
colnames(perf) <- c("Probability", "FPR", "TPR")

# Removing infinity value from data frame
perf <- perf[-1,]

perf21 <- perf
row.names(perf21) <- 1:nrow(perf21)
# Reshape the dataframe 
perf2 <- melt(perf, measure.vars = c("FPR", "TPR"))

ggplot(perf2, aes(Probability, value, colour=variable)) + 
  geom_line() + theme_bw()

  # You can see the cut-off by inspecting the perf dataframe

# model accuracy - Confusion Matrix
confusion.matrix(test_new$Dependent, pred_prob, threshold = 0.42)


  # We can optimise our model with respect to any of the metrics.
  # Sometime categorising or binning your continuous variable can also help 
  # in improving the overall model accuracy. Binning is particularly useful when 
  # there are lot of outliers in any continuous variables.

# Prediction on test set
pred_final <- predict(mod, newdata=test, type="response")
submit_final <- cbind(test$CAX_ID, pred_final)
colnames(submit_final) <- c("CAX_ID", "Dependent")
write.csv(submit_final, "Predictions.csv", row.names = F)



### Hey dont forget that this is a probabilistic model ####



# ||| Notes on the confusion matrix |||
confusion.matrix(test_new$Dependent, pred_prob, threshold = 0.42)

pred_prob # already have this, with the class probabilities of the test subjects
pred # a prediction object that uses the pred_prob vector

# Now we will try to do the other graphic
roc2 <- performance(pred, 'tnr', 'tpr')
# Create the data frame with values
desempeno <- as.data.frame(cbind(roc2@alpha.values[[1]], roc2@x.values[[1]], roc2@y.values[[1]]))
colnames(desempeno) <- c('Probability', 'tpr', 'tnr')
# Take the inf in the data frame 
desempeno <- desempeno[-1,]
desempe2 <- melt(desempeno, measure.vars = c('tpr', 'tnr'))
# This is the graph to see the sensitivity and specificity
ggplot(desempe2, aes(Probability, value, colour=variable)) + 
  geom_line() + theme_bw()





