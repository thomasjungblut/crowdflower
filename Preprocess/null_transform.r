# Example clean-up that does absolutely nothing
#
library(readr)
library(tm)

train_set <- read_csv("../Raw/train.csv")





write_csv(train_set, "../Processed/train_null.csv")

