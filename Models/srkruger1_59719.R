#LB SCORE = 0.59719

# 10 fold CV
# Accuracy  Kappa    Accuracy SD  Kappa SD  
# 0.681048  0.37856  0.01494788   0.03006305
require(readr)
require(stringr)
require(caret)
require(tau)
require(tm)

train <- read_csv("train.csv")
test <- read_csv("test.csv")

target <- ordered(train$median_relevance, labels=paste0("L", 1:4))

queryInTitle <- rep(FALSE, nrow(train))
queryCoverage <- rep(0, nrow(train))
qWords <- rep(0, nrow(train))
ptWords  <- rep(0, nrow(train))
pdWords <- rep(0, nrow(train))
for(i in 1:nrow(train))
{
    queryInTitle[i] <- grepl(train$query[i], train$product_title[i], ignore.case = TRUE)
    words <- unlist(str_split(train$query[i], " "))
    qWords[i] = length(words)
    ptWords[i] = length(unlist(str_split(train$product_title[i], " ")))
    pdWords[i] = length(unlist(str_split(train$product_description[i], " ")))
    sumCoverage = 0.0
    for(j in 1:length(words))
    {
        sumCoverage = sumCoverage + grepl(words[j], train$product_title[i], ignore.case = TRUE)
    }
    queryCoverage[i] <- sumCoverage / length(words)
}
myTrain <- data.frame(qit = factor(queryInTitle, labels=c("F", "T")), qc = queryCoverage,
                         qw = qWords, ptw = ptWords, pdw = pdWords)
write.csv(cbind(id = train$id, myTrain, median_relevance = train$median_relevance, 
                relevance_variance = train$relevance_variance), 
          "simpleTrain.csv", 
          row.names = FALSE)

queryInTitle <- rep(FALSE, nrow(test))
queryCoverage <- rep(0, nrow(test))
qWords <- rep(0, nrow(test))
ptWords  <- rep(0, nrow(test))
pdWords <- rep(0, nrow(test))
for(i in 1:nrow(test))
{
    queryInTitle[i] <- grepl(test$query[i], test$product_title[i], ignore.case = TRUE)
    words <- unlist(str_split(test$query[i], " "))
    qWords[i] = length(words)
    ptWords[i] = length(unlist(str_split(test$product_title[i], " ")))
    pdWords[i] = length(unlist(str_split(test$product_description[i], " ")))
    sumCoverage = 0.0
    for(j in 1:length(words))
    {
        sumCoverage = sumCoverage + grepl(words[j], test$product_title[i], ignore.case = TRUE)
    }
    queryCoverage[i] <- sumCoverage / length(words)
}
myTest <- data.frame(qit = factor(queryInTitle, labels=c("F", "T")), qc = queryCoverage,
                        qw = qWords, ptw = ptWords, pdw = pdWords)
write.csv(cbind(id = test$id, myTest), "simpleTest.csv", row.names = FALSE)

myTrain <- cbind(myTrain, q = factor(train$query))
myTest <- cbind(myTest, q = factor(test$query))


stopwords <- readLines(system.file("stopwords", "english.dat", package = "tm"))
ptWords <- tolower(train$product_title)
ptWords <- sapply(ptWords, FUN=remove_stopwords, words = stopwords, lines = TRUE)
ptWords <- removePunctuation(ptWords, preserve_intra_word_dashes = TRUE)
pt <- textcnt(ptWords,
                 n=1, 
                 split = "[[:space:]]", 
                 method="string", 
                 decreasing=TRUE,
                 lower = 0L)
commonPTWords <- head(names(pt), n=250)
trainPT <- as.data.frame(sapply(commonPTWords, FUN=grepl, ptWords, fixed=TRUE))
trainPT <- as.data.frame(sapply(trainPT[,], as.numeric))
names(trainPT) <- paste0("pt_", names(trainPT))

ptWords <- tolower(test$product_title)
ptWords <- sapply(ptWords, FUN=remove_stopwords, words = stopwords, lines = TRUE)
ptWords <- removePunctuation(ptWords, preserve_intra_word_dashes = TRUE)
testPT <- as.data.frame(sapply(commonPTWords, FUN=grepl, ptWords, fixed=TRUE))
testPT <- as.data.frame(sapply(testPT[,], as.numeric))
names(testPT) <- paste0("pt_", names(testPT))

pp <- preProcess(trainPT, "pca")
trainPT <- predict(pp, trainPT)
testPT <- predict(pp, testPT)

pdWords <- tolower(train$product_description)
pdWords <- sapply(pdWords, FUN=remove_stopwords, words = stopwords, lines = TRUE)
pdWords <- removePunctuation(pdWords, preserve_intra_word_dashes = TRUE)
pd <- textcnt(pdWords,
                 n=1, 
                 split = "[[:space:]]", 
                 method="string", 
                 decreasing=TRUE,
                 lower = 0L)
commonPDWords <- head(names(pd), n=250)
trainPD <- as.data.frame(sapply(commonPDWords, FUN=grepl, pdWords, fixed=TRUE))
trainPD <- as.data.frame(sapply(trainPD[,], as.numeric))
names(trainPD) <- paste0("pd_", names(trainPD))

pdWords <- tolower(test$product_description)
pdWords <- sapply(pdWords, FUN=remove_stopwords, words = stopwords, lines = TRUE)
pdWords <- removePunctuation(pdWords, preserve_intra_word_dashes = TRUE)
testPD <- as.data.frame(sapply(commonPDWords, FUN=grepl, pdWords, fixed=TRUE))
testPD <- as.data.frame(sapply(testPD[,], as.numeric))
names(testPD) <- paste0("pd_", names(testPD))

pp <- preProcess(trainPD, "pca")
trainPD <- predict(pp, trainPD)
testPD <- predict(pp, testPD)

save.image("work.RData")
load("work.RData")

gc(reset=T)

set.seed(65812)
fc <- trainControl(method = "repeatedCV",
                      number = 10, repeats = 1, verboseIter = TRUE, 
                      returnResamp = "all", classProbs = TRUE)
tGrid <- expand.grid(interaction.depth = 9, shrinkage = 0.1, n.trees = 50)

model <- train(x = cbind(myTrain, trainPT, trainPD), y = target, method = "gbm", trControl = fc, 
                  tuneGrid = tGrid, metric = "Accuracy") #
model
confusionMatrix.train(model)

submit <- data.frame(id = test$id, prediction = as.integer(predict(model, cbind(myTest, testPT, testPD))))
write.csv(submit, "submit.csv", row.names = FALSE)