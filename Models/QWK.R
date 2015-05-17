#Evaluation metric function to be used with caret's trainControl
#requires the Metrics package
QWK <- function (data, lev = NULL, model = NULL) 
{
    real <- as.integer(data$obs)
    preds <- as.integer(data$pred)
    
    out <- ScoreQuadraticWeightedKappa(real, preds, 1, 4)
    names(out) <- c("QWK")
    out
}