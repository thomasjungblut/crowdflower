
require(readr)
require(tm)

rm(list=ls())

train_set <- read_csv("../Raw/train.csv")
test_set  <- read.csv("../Raw/test.csv")

scrub_data <- function(doc_vector) {
    print("Scrub")
    corpus <- VCorpus(VectorSource(doc_vector))
    # 12382 -> number
    corpus <- tm_map(corpus, 
        content_transformer(function(x) gsub("\\d+", "number", x)))
    # High freq. words such as "a", "the", etc removed
    corpus <- tm_map(corpus, removeWords, stopwords("english"))
    # Sandals -> Sandal,  memory -> memori
    corpus <- tm_map(corpus, stemDocument)
    return(vapply(corpus, as.character, FUN.VALUE="", USE.NAMES=FALSE))
}

train_set$query <- scrub_data(train_set$query)
train_set$product_title <- scrub_data(train_set$product_title)
# Drop description, has for now too much and too dirty data
train_set$product_description <- NULL

#
test_set$query <- scrub_data(test_set$query)
test_set$product_title <- scrub_data(test_set$product_title)

#----------------------------------------------------------------------------
write_csv(train_set, "../Processed/train_scrubbed.csv")
write_csv(test_set,  "../Processed/test_scrubbed.csv")

