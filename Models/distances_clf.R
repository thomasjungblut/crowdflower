#LB SCORE = ???

# 10 fold CV
# QWK        QWK SD    
# 0.53         0.012
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

#train <- train[,!(names(train) %in% dropCols)]
#test <- test[,!(names(test) %in% dropCols)]

testIds <- test$id
target <- as.factor(train$outcome)

# remove id
train <- train[, -1]
test <- test[, -1]

#remove outcome
train <- train[, -dim(train)[2]]


set.seed(65812)
fc <- trainControl(method = "repeatedCV", summaryFunction=QWK,
                      number = 3, repeats = 1, verboseIter = TRUE, 
                      returnResamp = "all", classProbs = TRUE)


#nnetGrid <- expand.grid(size = c(seq(20,100,10)), decay = c(0))
#model <- train(x = train, y = target, method = "nnet", trControl = fc, metric = "QWK", tuneGrid = nnetGrid, MaxNWts=100000) 

rfGrid <- expand.grid(mtry = c(13:20)) #16
model <- train(x = train, y = target, method = "rf", trControl = fc, metric = "QWK", tuneGrid = rfGrid) 

model
#confusionMatrix.train(model)

prediction <- predict(model, test)

submit <- data.frame(id = testIds, prediction = as.integer(prediction))
write.csv(submit, "Distances/Submission/submit.csv", row.names = FALSE)