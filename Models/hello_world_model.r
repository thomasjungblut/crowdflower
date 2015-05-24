#-------------------------------------------------------------------------
require(readr)
require(tm)
require(randomForest)

#-------------------------------------------------------------------------
training_set <- read_csv("../Processed/train_scrubbed.csv")
test_set     <- read_csv("../Processed/test_scrubbed.csv")

#-------------------------------------------------------------------------
train_feature_set <- read_csv("../Processed/train_features.csv")
test_feature_set  <- read_csv("../Processed/test_features.csv")

train_feature_set$y <- factor(training_set$median_relevance)
model1 <- randomForest(y~., train_feature_set, do.trace=FALSE)

prediction <- predict(model1, test_feature_set)

print(model1)
print(table(train_feature_set$y)/sum(table(train_feature_set$y)))
print(table(prediction)/sum(table(prediction)))

#-------------------------------------------------------------------------
# Construct the submission
submission <- data.frame(
    id = test_set$id,
    prediction = prediction
)
write_csv(submission, "toy_submission.csv")

#-------------------------------------------------------------------------
