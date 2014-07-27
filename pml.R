library(caret)

train <- read.csv("pml-training.csv")
train <- train[,-(1:6)]
columns <- colSums(is.na(train) | train=="") <= nrow(train)*0.2
train <- train[,columns]

preProc <- preProcess(train[,-ncol(train)], method="pca", thresh=0.9)
trainPC <- predict(preProc, train[,-ncol(train)])

modFit <- train(train$classe ~ ., data=trainPC, method="rf", trControl=trainControl(method = 'cv', number=5))

pml_write_files = function(x) {
	n = length(x)
	for (i in 1:n) {
		filename = paste0("problem_id_",i,".txt")
		write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
	}
}

test <- read.csv("pml-testing.csv")
test <- test[,-(1:6)]
test <- test[,columns]
testPC <- predict(preProc, test[,-ncol(test)])
outcomes <- predict(modFit, testPC)
pml_write_files(outcomes)
