
require(readr)
require(tm)

rm(list=ls())

args <- commandArgs(trailingOnly=TRUE)

in_name  <- args[1]
out_name <- args[2]

data_set <- read_csv(in_name)

scrub_data <- function(doc_vector) {
    print("Scrub")
    gg <- gsub("\\s", " ", doc_vector)
    gg <- tolower(gg)
    if (0) {
        # This takes care of most of the CSS
        gg <- gsub("\\.[a-z]+\\s", " ", gg)
        gg <- gsub("\\b[a-z]+\\.[a-z]+\\b", " ", gg)
        gg <- gsub("\\b[a-z-]+:[^;]+;", " ", gg)
        gg <- gsub("\\b(li|ul|hr|h1|h2|h3|h4|h5|body|p|div|table)\\b", " ", gg)
        gg <- gsub("'", "", gg)
        gg <- gsub(":", " ", gg)
        gg <- gsub("\\.", " ", gg)
        # This takes care of most of the HTML
        gg <- gsub("<[a-z]+>", "", gg)
        gg <- gsub("</[a-z]+>", "", gg)
    }
    # Remove all unicode and other nonsense
    gg <- gsub("[^a-z 0-9-]", "#", gg)
    #gg <- gsub("-", " ", gg)
    gg <- gsub("#", "", gg)
    gg <- gsub("\\s+", " ", gg)
    # Remove possible space at the begin of a sentence.
    gg <- gsub("^\\s", "", gg)
    return(gg)
}

#----------------------------------------------------------------------------
data_set$query         <- scrub_data(data_set$query)
data_set$product_title <- scrub_data(data_set$product_title)
data_set$product_description <- scrub_data(data_set$product_description)

#----------------------------------------------------------------------------
write_csv(data_set, out_name)

# vi: spell spl=en
