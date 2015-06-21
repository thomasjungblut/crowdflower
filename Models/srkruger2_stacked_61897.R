require(tau)
require(tm)
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

#Save predictions for later stacking
testPreds <- cbind(predict(model, cbind(myTest, testBlood), type="prob"))
write.csv(testPreds, "modelTestPreds.csv", row.names = FALSE)
trainPreds <- cbind(predict(model, cbind(myTrain, trainBlood), type="prob"))
write.csv(trainPreds, "modelTrainPreds.csv", row.names = FALSE)

#BOW on product_title
# stopwords <- readLines(system.file("stopwords", "english.dat", package = "tm"))
ptWords <- tolower(train$product_title)
# ptWords <- sapply(ptWords, FUN=remove_stopwords, words = stopwords, lines = TRUE)
ptWords <- removePunctuation(ptWords, preserve_intra_word_dashes = TRUE)
pt <- textcnt(ptWords,
              n=1, 
              split = "[[:space:]]", 
              method="string", 
              decreasing=TRUE,
              lower = 20L)
commonPTWords <- head(names(pt), n=150)
trainPT <- as.data.frame(sapply(commonPTWords, FUN=grepl, ptWords, fixed=TRUE))
trainPT <- as.data.frame(sapply(trainPT[,], as.numeric))
names(trainPT) <- paste0("pt_", names(trainPT))
ptWords <- tolower(test$product_title)
# ptWords <- sapply(ptWords, FUN=remove_stopwords, words = stopwords, lines = TRUE)
ptWords <- removePunctuation(ptWords, preserve_intra_word_dashes = TRUE)
testPT <- as.data.frame(sapply(commonPTWords, FUN=grepl, ptWords, fixed=TRUE))
testPT <- as.data.frame(sapply(testPT[,], as.numeric))
names(testPT) <- paste0("pt_", names(testPT))

#Build the stacked model
gc()
set.seed(96583)
fc <- trainControl(method = "repeatedCV", summaryFunction=QWK,
                   number = 3, repeats = 10, verboseIter = TRUE, 
                   returnResamp = "all")
tGrid <- expand.grid(decay = 0.1)

stack <- train(x = cbind(trainPreds, trainPT), y = target, method = "multinom", trControl = fc, 
            tuneGrid = tGrid, metric = "QWK")
stack
confusionMatrix.train(stack)

#Submit using the stacked model
submit <- data.frame(id = test$id, prediction = as.integer(predict(stack, cbind(testPreds, testPT))))
write.csv(submit, "submit.csv", row.names = FALSE)


