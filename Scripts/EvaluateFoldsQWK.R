require(readr)
require(caret)
require(Metrics)
source("../Models/QWK.R")

args <- commandArgs(trailingOnly = TRUE)
print(args)

modelCvOutputPath =  "../Models/NaiveBayes/CvFoldsOutput/"
cvOutputPath =  "../ValidationFolds/"
# modelCvOutputPath <- args[0]
# CvOutputPath <- args[1]



CVQKW <- function (cvOutputPath, modelCvOutputPath) 
{
  modelOutputFolds <- list.files(path = modelCvOutputPath, pattern = "*test.csv",
                                 full.names = FALSE, include.dirs = FALSE)
  foldKappa <- rep(0, length(modelOutputFolds))
  
  for(i in 1:length(modelOutputFolds)) {  
    prediction <- as.numeric(read_csv(paste(modelCvOutputPath, modelOutputFolds[i], sep = ""))$predictedClass)
    outcome <- as.numeric(read_csv(paste(cvOutputPath, modelOutputFolds[i], sep = ""))$median_relevance)
    
    if(length(prediction) != length(outcome)) {
      stop(paste("length in files", modelOutputFolds[i], "didn't match"))  
    }
    
    df <- as.data.frame(cbind(outcome, prediction))
    colnames(df) <- c("obs", "pred")
    qwk <- QWK(df)
    foldKappa[i] <- qwk
    
    # useful?
    # confusionMatrix(prediction, outcome)
  }
  
  return(foldKappa)
}

foldKappa <- CVQKW(cvOutputPath, modelCvOutputPath)

print(foldKappa)
summary(foldKappa)