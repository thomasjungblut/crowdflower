require(readr)
require(tm)
require(stringr)

#-------------------------------------------------------------------------
count_words <- function(words) {
    return(vapply(words, length, FUN.VALUE=1))
}

#-------------------------------------------------------------------------

if (1) {
    args <- commandArgs(trailingOnly=TRUE)
    in_name  <- args[1]
    out_name <- args[2]
} else {
    # To be able to easily debug
    in_name  <- "../Processed/train_scrubbed.csv"
    out_name <- "../Processed/train_features.csv"
}
data_set <- read_csv(in_name)

query_words <- str_split(data_set$query, "\\s+")
title_words <- str_split(data_set$product_title, "\\s+")

title_word_count <- count_words(title_words)
query_word_count <- count_words(query_words)

all_words <- mapply(c, query_words, title_words)

intersection <- mapply(intersect, query_words, title_words)

hit_rate <- count_words(intersection)/count_words(query_words)

feature_set <- data.frame(
    X1 = count_words(intersection),
    X2 = count_words(query_words),
    X3 = count_words(title_words),
    X4 = hit_rate
)

write_csv(feature_set, out_name)




#==============================================================================
if (0) {
corpus <- VCorpus(VectorSource(data_set$query))
# write_csv(data_set, out_name")

dtm <- DocumentTermMatrix(corpus,
           control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE),
                          stopwords = TRUE))
}
