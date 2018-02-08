## Bats_UY A machine learning algorithm for bat identification in Uruguay
##    Version 2.0
##    Copyright (C) 2017  German Botto Nuñez - Universidad de la Republica
##
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
##    germanbotto@gmail.com
##    https://sites.google.com/site/germanbotto/home

library(gamlss)
library(gamlss.add)
library(dplyr)
library(randomForest)
library(caret)
library(kernlab)
library(nnet)
library(neuralnet)

# load(raw_data.RData)

##### DATABASE SUBSETTING ######
raw_data$ID1 <- as.character(row.names(raw_data))
set.seed(1)
raw_data$group <- ifelse(raw_data$ID1
                         %in%
                           subset(raw_data %>%
                                    group_by(SP) %>%
                                    sample_frac(size = 1/3),
                                  select = "ID1")$ID1,
                         "VarSelection", "Model")
table(raw_data$group)
# save(raw_data, file = "Raw_data_groups.RData")
# load(file = "Raw_data_groups.RData")


##### FOWARD SELECTION #####
mod_fow <- as.list(NULL)
a1 <- c("SP")
for (i in (1:72)){
  a1 <- c(a1,labels(var.sel1$importance)[[1]] [which(order(var.sel1$importance, decreasing = TRUE)==i)])
  mod_fow[[i]] <- randomForest(SP~., data = subset(raw_data, raw_data$group == "VarSelection", select = a1))
}

err1 <- c()
for(k in (1:72)){
  err1 <- c(err1,1-(sum(diag(mod_fow[[k]]$confusion[1:8,1:8])))/(sum(mod_fow[[k]]$confusion[1:8,1:8])))
}

##### MODEL SELECTION AND TESTING #####
vars <- row.names(mod_fow[[12]]$importance) ## predictors from the variable selection process (best performing model)
SP <- subset(raw_data, raw_data$group == "Model", select = "SP") 
bats <- raw_data[which(raw_data$group == "Model") , which(names(raw_data) %in% vars)]
scaled <- as.data.frame(scale(bats, center = apply(bats, 2, min), scale = apply(bats, 2, max) - apply(bats, 2, min))) ## [0,1] scaling of the numeric variables
bats_scale <- cbind(scaled, SP) ## Dataframe for the neural networks
bats <- cbind(bats, SP) ## Dataframe for the SVM and RF
set.seed(101)
split_samples <- createDataPartition(SP$SP, p = .5, times = 100) ## 100 independent splits of the data
f1 <- as.formula(paste("SP", " ~ ", paste(vars, collapse = " + "))) ## formula for SVM and RF
err_rates_RF <- as.vector(NULL) ## Vector for global error estimates 
err_rates_SVM <- as.vector(NULL)
err_rates_ANN <- as.vector(NULL)


for(i in 1:100){
  train_sc <- bats_scale[split_samples[[i]], ]
  test_sc <- bats_scale[-split_samples[[i]], ]
  train <- bats[split_samples[[i]], ]
  test <- bats[-split_samples[[i]], ]
  
  multi_rf<-randomForest(f1, data = train,importance=T, proximity=T, na.action=na.omit, ntree=5000)
  err_rates_RF <- c(err_rates_RF, prop.table(table(as.character(predict(multi_rf, newdata=test))==as.character(test$SP)))[[1]])
  
  multi_svm <- ksvm(f1, data = train,kernel = "rbfdot", prob.model = TRUE, C = 1)
  err_rates_SVM <- c(err_rates_SVM, prop.table(table(as.character(predict(multi_svm, newdata=test,type="response"))==as.character(test$SP)))[[1]]) 
  
  ann <- nnet(f1, data = train_sc, size = 10)
  err_rates_ANN <- c(err_rates_SVM, prop.table(table(test_sc$SP== predict(ann, newdata = test_sc, type = "class")))[[1]])
}

boxplot(err_rates_RF, err_rates_SVM, err_rates_ANN, names = c("RF", "SVM", "ANN"))

##### TRESHOLD ESTIMATION ######
multi_rf1<-randomForest(f1, data = train,importance=T, proximity=T, na.action=na.omit, ntree=5000) ## OOB error=5.8%
prop.table(table(as.character(predict(multi_rf1, newdata=test))==as.character(test$SP)))[[1]] ## honest error=6.9%

pred.test <- cbind(test, predict(multi_rf1, newdata = test, type = "prob"))
pred.test$pred.sp <- as.character(names(pred.test)[13 + apply(pred.test[,14:21], 1, which.max)])
pred.test$pred.prob <- apply(pred.test[,14:21], 1, max)

tresh.error <- rep.int(NA, times = 100)

for(j in 1:100){
  tresh.error[j] <- sum(ifelse(pred.test$pred.prob >= seq(from = 0.01, to = 1, by = 0.01)[j], 
                               ifelse(as.character(pred.test$SP) == pred.test$pred.sp, 0, 1), 0.5))
}
plot(tresh.error, type = "l")
rev(seq(from = 0.01, to = 1, by = 0.01))[which.min(rev(tresh.error))] ## the last global minimum 0.43

# save(multi_rf1, file = "multi_rf1.RData")