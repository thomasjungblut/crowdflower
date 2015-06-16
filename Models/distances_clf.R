#LB SCORE = ???

# 10 fold CV
# QWK        QWK SD    
# 0.42         0.26
source("QWK.R")
require(caret)
require(Metrics)
require(readr)
require(randomForest)
require(MASS)

train <- read_csv("../Processed/distances_features.csv")
test <- read_csv("../Processed/distances_features_test.csv")

# not so significant features
dropCols <- c( "wordngcount2",
               "leftngcount2",
               "wordngjacc2",
               "wordngcount3",
               "leftngcount3",
               "wordngjacc3", 
               "wordngcount4",
               "leftngcount4",
               "wordngjacc4",
               "wordngcount5",
               "leftngcount5",
               "wordngjacc5" )

train <- train[,!(names(train) %in% dropCols)]
test <- test[,!(names(test) %in% dropCols)]

testIds <- test$id
target <- as.factor(train$outcome)

# remove id
train <- train[, -1]
test <- test[, -1]

#remove outcome
train <- train[, -dim(train)[2]]


set.seed(65812)
fc <- trainControl(method = "repeatedCV", summaryFunction=QWK,
                      number = 10, repeats = 3, verboseIter = TRUE, 
                      returnResamp = "all", classProbs = TRUE)

grid <-  expand.grid(mtry=c(13))

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

model <- train(x = train, y = target, method = "rf", trControl = fc, metric = "QWK",
               tuneGrid = grid, nTree = 500) 
model
#confusionMatrix.train(model)

prediction <- predict(model, test)

submit <- data.frame(id = testIds, prediction = as.integer(prediction))
write.csv(submit, "submit.csv", row.names = FALSE)