require(readr)
require(stringr)

#Use scrubbed data
train <- read_csv("../../Processed/train_scrubbed.csv")
test <- read_csv("../../Processed/test_scrubbed.csv")

#Build corpora and write 'em to file
A <- unique(c(train$product_title, train$product_description, test$product_title, test$product_description))
B <- c(train$query, test$query)
C <- c(train$product_title, test$product_title)
fileConn <- file("A.txt")
writeLines(A, fileConn)
close(fileConn)
fileConn <- file("B.txt")
writeLines(B, fileConn)
close(fileConn)
fileConn <- file("C.txt")
writeLines(C, fileConn)
close(fileConn)

### Run GDBuilder C# Code ###

#Load the dictionary
dict <- as.data.frame(read_csv("sample_dict.csv"))
#Apply minimum count to reduce the size quite a bit
MIN_COUNT = 20
dict <- dict[dict$co >= MIN_COUNT, ]
#Apply min word length
MIN_WORD_LEN = 2
dict <- dict[str_length(dict$word1) >= MIN_WORD_LEN & 
                 (str_length(dict$word2) >= MIN_WORD_LEN | dict$word2 == ""), ]

#Read the Distances
fileConn <- file("distances.txt")
distances <- as.numeric(readLines(fileConn))
close(fileConn)
#and plot against target
plot(factor(train$median_relevance), distances[1:nrow(train)])

#See http://en.wikipedia.org/wiki/Normalized_Google_distance
#Uses the dictionary produced by the GDBuilder
GoogleDistance <- function(dictionary, word1, word2, size)
{
    gd <- Inf
    word12 <- sort(c(word1, word2), decreasing = FALSE)
    word1 <- word12[1]
    word2 <- word12[2]
    if(sum(dictionary$word1 == word1 & dictionary$word2 == "") == 1 &
           sum(dictionary$word1 == word2 & dictionary$word2 == "") == 1 &
           sum(dictionary$word1 == word1 & dictionary$word2 == word2) == 1)
    {
        lw1 <- log(dictionary[dictionary$word1 == word1 & dictionary$word2 == "", "co"])
        lw2 <- log(dictionary[dictionary$word1 == word2 & dictionary$word2 == "", "co"])
        lw12 <- log(dictionary[dictionary$word1 == word1 & dictionary$word2 == word2, "co"])
        gd <- (max(lw1, lw2) - lw12) / (log(size) - min(lw1, lw2))
    }
    gd
}

GoogleDistance(dict, "accent", "pillow", length(A))
GoogleDistance(dict, "samsung", "phone", length(A))
GoogleDistance(dict, "appl", "phone", length(A))
GoogleDistance(dict, "iphon", "appl", length(A))
GoogleDistance(dict, "ipad", "appl", length(A))
GoogleDistance(dict, "nexus", "googl", length(A))
GoogleDistance(dict, "white", "appl", length(A))
GoogleDistance(dict, "white", "samsung", length(A))
GoogleDistance(dict, "kitchen", "style", length(A))
GoogleDistance(dict, "kitchen", "electric", length(A))
GoogleDistance(dict, "kitchen", "steel", length(A))
GoogleDistance(dict, "kitchen", "counter", length(A))

#See http://en.wikipedia.org/wiki/Normalized_Google_distance
#Uses the dictionary produced by the GDBuilder
FindClosestWords <- function(dictionary, word, size, limit = 0)
{
    if(sum(dictionary$word1 == word) == 0 | sum(dictionary$word2 == word) == 0)
        return(NULL)
    
    occursw1 <- dictionary$word1 == word & dictionary$word2 != ""
    occursw2 <- dictionary$word2 == word & dictionary$word1 != ""
    relatedWords <- c(dictionary$word2[occursw1], dictionary$word1[occursw2])
    if(limit > 0)  
        relatedWords <- head(relatedWords[order(-c(dictionary$co[occursw1], dictionary$co[occursw2]))], limit)
    
    distances <- sapply(relatedWords, FUN = GoogleDistance, dictionary = dict, word2 = word, size = size)
    relatedWords <- data.frame(word = relatedWords, distance = distances)
    relatedWords <- relatedWords[order(relatedWords$distance), ]
    relatedWords
}

FindClosestWords(dict, "samsung", length(A), 10)
FindClosestWords(dict, "appl", length(A), 10)
FindClosestWords(dict, "kitchen", length(A), 10)
FindClosestWords(dict, "number", length(A), 10)
FindClosestWords(dict, "googl", length(A), 10)
FindClosestWords(dict, "electric", length(A), 10)
FindClosestWords(dict, "brand", length(A), 10)
FindClosestWords(dict, "wifi", length(A), 10)

#Uses the dictionary produced by the GDBuilder
#Not working well
PhraseDistance <- function(dictionary, phrase1, phrase2, size)
{
    words1 <- unlist(str_split(phrase1, pattern="[[:space:]]"))
    words2 <- unlist(str_split(phrase2, pattern="[[:space:]]"))
    
    numDistances = 0
    sumDistances = 0
    for(i in 1:length(words1))
    {
        w1 <- words1[i]
        if(w1 == "")
            next
        
        for(j in 1:length(words2))
        {
            w2 <- words2[j]
            if(w2 == "")
                next
            
            if(w1 != w2)
            {
                
                d = GoogleDistance(dictionary, w1, w2, size)
                if(d != Inf)
                {
                    numDistances = numDistances + 1
                    sumDistances = sumDistances + d
                }
            }
        }
    }
    if(numDistances == 0)
        return(Inf)
    if(sumDistances == 0)
        return(0)
    
    sumDistances / numDistances
}

PhraseDistance(dict, "cell phon", "samsung galaxi not", length(A))
PhraseDistance(dict, "cell phon", "wifi router", length(allText))
PhraseDistance(dict, "cell phon", "wireless number modem", length(A))
PhraseDistance(dict, "playstation games bundl", "soni", length(A))
PhraseDistance(dict, "console game", "nintendo wii", length(A))
PhraseDistance(dict, "boyfriend jean", "how make american quilt (dvd)", length(A))
PhraseDistance(dict, "boyfriend jean", "levi", length(A))
PhraseDistance(dict, "boyfriend jean", "zipper levi style", length(A))



