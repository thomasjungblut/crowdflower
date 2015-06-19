require(readr)

training_set <- read_csv("../Processed/train_scrubbed.csv")
test_set     <- read_csv("../Processed/test_scrubbed.csv")
training_set$query <- factor(training_set$query)
test_set$query     <- factor(test_set$query, 
                             levels=levels(training_set$query))

test_set$query_id     <- as.integer(test_set$query)
training_set$query_id <- as.integer(training_set$query)

ttt <- data.frame(
    query_id <- as.integer(training_set$query),
    rating   <- training_set$median_relevance
)
max_ttt <- aggregate(rating ~ query_id, ttt, max)
min_ttt <- aggregate(rating ~ query_id, ttt, min)
colnames(max_ttt) <- c("query_id", "max_rating")
colnames(min_ttt) <- c("query_id", "min_rating")
ratings <- merge(max_ttt, min_ttt)

test_features <- data.frame(
    maxr = ratings[test_set$query_id,"max_rating"],
    minr = ratings[test_set$query_id,"min_rating"]
)

training_features <- data.frame(
    maxr = ratings[training_set$query_id,"max_rating"],
    minr = ratings[training_set$query_id,"min_rating"]
)

write_csv(training_features, "../Processed/train_minmaxr.csv")
write_csv(test_features,     "../Processed/test_minmaxr.csv")

