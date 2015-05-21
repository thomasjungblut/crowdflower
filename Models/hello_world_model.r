require(readr)
require(tm)
require(randomForest)

#-------------------------------------------------------------------------

training_set <- read_csv("../Processed/train_scrubbed.csv")

#-------------------------------------------------------------------------
train_feature_set <- read_csv("../Processed/train_features.csv")
test_feature_set  <- read_csv("../Processed/test_features.csv")

train_feature_set$y <- factor(training_set$median_relevance)
fit1 <- randomForest(y~., train_feature_set, do.trace=FALSE)

print(fit1)

