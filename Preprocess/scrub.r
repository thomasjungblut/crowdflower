
require(readr)
require(tm)

rm(list=ls())

args <- commandArgs(trailingOnly=TRUE)

in_name  <- args[1]
out_name <- args[2]

data_set <- read_csv(in_name)

scrub_data <- function(doc_vector) {
    print("Scrub")
    corpus <- VCorpus(VectorSource(doc_vector))
    # Don't care for punctuation
    corpus <- tm_map(corpus, removePunctuation)
    # Everything in lower case
    corpus <- tm_map(corpus, content_transformer(tolower))
    # 12382 -> number
    corpus <- tm_map(corpus, 
        content_transformer(function(x) gsub("\\b\\d+\\b", "number", x)))
    # High freq. words such as "a", "the", etc removed
    corpus <- tm_map(corpus, removeWords, stopwords("english"))
    # Sandals -> Sandal,  memory -> memori
    corpus <- tm_map(corpus, stemDocument)
    # Remove unnecessary spacing
    corpus <- tm_map(corpus, stripWhitespace)
    return(vapply(corpus, as.character, FUN.VALUE="", USE.NAMES=FALSE))
}

#----------------------------------------------------------------------------
data_set$query         <- scrub_data(data_set$query)
data_set$product_title <- scrub_data(data_set$product_title)
# Drop description, has for now too much and too dirty data
data_set$product_description <- NULL

#----------------------------------------------------------------------------
write_csv(data_set, out_name)

# vi: spell spl=en
