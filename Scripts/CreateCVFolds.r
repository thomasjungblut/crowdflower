require(readr)
require(caret)

'%nin%' <- Negate('%in%')

numFolds <- 10

train <- read_csv("../Raw/train.csv")

y = as.factor(train$median_relevance)

folds <- createFolds(y, numFolds, list = TRUE, returnTrain = FALSE)
testFolds <- lapply(folds, function(ind, dat) dat[ind,], dat = train)

for(i in 1:length(testFolds)) {  
  foldTest <- as.data.frame(testFolds[i])
  colnames(foldTest) <- colnames(train)  
  foldTrain <- train[rownames(train) %nin% unlist(folds[i]),]
  
  trainPath <- paste("../ValidationFolds/", i, "-train.csv", sep = "")
  testPath <- paste("../ValidationFolds/", i, "-test.csv", sep = "")
  
  write_csv(foldTrain, trainPath)
  write_csv(foldTest, testPath)
}