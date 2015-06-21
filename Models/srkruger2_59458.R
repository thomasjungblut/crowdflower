require(readr)
require(stringr)

train <- read_csv("../Processed/train_scrubbed.csv")
test <- read_csv("../Processed/test_scrubbed.csv")

trainBlood <- read_csv("../Processed/distances_features.csv")
trainBlood <- trainBlood[, -which(names(trainBlood) == "outcome")]
testBlood <- read_csv("../Processed/distances_features_test.csv")

target <- ordered(train$median_relevance, labels=paste0("L", 1:4))

qInT <- rep(0, nrow(train))
qInD <- rep(0, nrow(train))
qWords <- rep(0, nrow(train))
ptWords  <- rep(0, nrow(train))
pdWords <- rep(0, nrow(train))
for(i in 1:nrow(train))
{    
    words <- unlist(str_split(train$query[i], " "))
    qWords[i] = length(words)
    ptWords[i] = length(unlist(str_split(train$product_title[i], " ")))
    pdWords[i] = length(unlist(str_split(train$product_description[i], " ")))
    sumQT = 0.0
    sumQD = 0.0
    for(j in 1:length(words))
    {
        sumQT = sumQT + grepl(words[j], train$product_title[i], ignore.case = TRUE)
        sumQD = sumQD + grepl(words[j], train$product_description[i], ignore.case = TRUE)        
    }    
    qInT[i] <- sumQT / length(words)
    qInD[i] <- sumQD / length(words)
}
myTrain <- data.frame(qInT = qInT, qInD = qInD,
                      qInT_qWords = qInT * qWords,
                      qInT_ptWords = qInT * ptWords, qInT_pdWords = qInT * pdWords,
                      ptw = ptWords, pdw = pdWords)

qInT <- rep(0, nrow(test))
qInD <- rep(0, nrow(test))
qWords <- rep(0, nrow(test))
ptWords  <- rep(0, nrow(test))
pdWords <- rep(0, nrow(test))
for(i in 1:nrow(test))
{    
    words <- unlist(str_split(test$query[i], " "))
    qWords[i] = length(words)
    ptWords[i] = length(unlist(str_split(test$product_title[i], " ")))
    pdWords[i] = length(unlist(str_split(test$product_description[i], " ")))
    sumQT = 0.0
    sumQD = 0.0
    for(j in 1:length(words))
    {
        sumQT = sumQT + grepl(words[j], test$product_title[i], ignore.case = TRUE)
        sumQD = sumQD + grepl(words[j], test$product_description[i], ignore.case = TRUE)        
    }    
    qInT[i] <- sumQT / length(words)
    qInD[i] <- sumQD / length(words)
}
myTest <- data.frame(qInT = qInT, qInD = qInD,
                     qInT_qWords = qInT * qWords,
                     qInT_ptWords = qInT * ptWords, qInT_pdWords = qInT * pdWords,
                     ptw = ptWords, pdw = pdWords)

myTrain <- cbind(myTrain, q = factor(train$query))
myTest <- cbind(myTest, q = factor(test$query))

require(caret)
require(Metrics)
source("QWK.R")

set.seed(65812)
fc <- trainControl(method = "repeatedCV", summaryFunction=QWK,
                   number = 3, repeats = 3, verboseIter = TRUE, 
                   returnResamp = "all")
tGrid <- expand.grid(interaction.depth = 6, shrinkage = 0.1, n.trees = 30)

model <- train(x = cbind(myTrain, trainBlood), y = target, method = "gbm", trControl = fc, 
               tuneGrid = tGrid, metric = "QWK")
model
confusionMatrix.train(model)

submit <- data.frame(id = test$id, prediction = as.integer(predict(model, cbind(myTest, testBlood))))
write.csv(submit, "submit.csv", row.names = FALSE)
