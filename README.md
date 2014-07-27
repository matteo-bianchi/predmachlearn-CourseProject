### Practical Machine Learning - Course Project

The aim of this course project is to build a predictor that is able to deduct if a person is performing barbell lifts correctly or incorrectly based on the measurements gathered by a set of devices such as accelerometers and gyroscopes worn on different parts of the body to track their movements.
The predictor must be trained on a collection of sample data recorded from people doing the exercise correctly and incorrectly and tagged with the outcome of the prediction.

First of all we load the training set and take a look at the names and types of the available variables.

<!-- -->
    train <- read.csv("pml-training.csv")
	names(train)

`classe` is the outcome that we must predict.
We then exclude from the dataset the first six columns since they seem to be not so useful to make the prediction: column `X` seems to be simply the index of the observation, `user_name` is the name of the person doing the exercise, the next three variables seem to relate to the moment in which the exercise has been done, the `new_window` values could have something to do with the begin/end of a workout session (perhaps even the `num_window` column plays no role in the prediction).

<!-- -->
	train <- train[,-(1:6)]
Done that we remain with 154 columns. A lot of these though have no value at all or have very few values. We therefore get rid of columns with more than 20% missing values.

<!-- -->
	columns <- colSums(is.na(train) | train=="") <= nrow(train)*0.2
	train <- train[,columns]

Now we are left with 53 numeric variables and the last column which contains the class of the observations that we must predict. We proceed searching for highly correlated variables that can be replaced with a simpler structure. For doing this we take advantage of Principal Component Analysis and use it to generate a lower number of variables that still retain 90% of the variance of the data.

<!-- -->
	preProc <- preProcess(train[,-ncol(train)], method="pca", thresh=0.9)
	trainPC <- predict(preProc, train[,-ncol(train)])
	# -ncol(train) leaves out the last column which contains the class

At this point only 19 variables are left and we can train our predictor. Since the whole experiment doesn't need to be reproducible the random seed won't be set to any specific value. A random forest model is built using 5-fold cross-validation to estimate the out of sample error.

<!-- -->
	modFit <- train(train$classe ~ .,
	                data=trainPC,
	                method="rf",
	                trControl=trainControl(method='cv', number=5))

The resulting model achieves about 98% accuracy:

<!-- -->
	Random Forest 
	
	19622 samples
	   18 predictors
	    5 classes: 'A', 'B', 'C', 'D', 'E' 
	
	No pre-processing
	Resampling: Cross-Validated (5 fold) 
	
	Summary of sample sizes: 15697, 15698, 15698, 15697, 15698 
	
	Resampling results across tuning parameters:
	
	  mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
	  2     0.979     0.974  0.00319      0.00403 
	  10    0.975     0.968  0.00375      0.00475 
	  19    0.97      0.962  0.00444      0.00562 
	
	Accuracy was used to select the optimal model using  the largest value.
	The final value used for the model was mtry = 2.

<!-- -->
	Call:
	 randomForest(x = x, y = y, mtry = param$mtry) 
	               Type of random forest: classification
	                     Number of trees: 500
	No. of variables tried at each split: 2
	
	        OOB estimate of  error rate: 1.58%
	Confusion matrix:
	     A    B    C    D    E class.error
	A 5548   12   13    6    1 0.005734767
	B   34 3724   34    0    5 0.019225705
	C    8   36 3359   18    1 0.018410286
	D    8    3  106 3096    3 0.037313433
	E    0    5    8   10 3584 0.006376490

Since it is built on top of the variables computed by the PCA its tree contains nodes that refer to constructed variables named `PC1`, `PC2`, etc., and so it doesn't make much sense looking at its internal structure.

We can now predict the outcomes of the given test set after the same processing has been applied even to these observations.

<!-- -->
	test <- read.csv("pml-testing.csv")
	test <- test[,-(1:6)]
	test <- test[,columns]
	testPC <- predict(preProc, test[,-ncol(test)])
	outcomes <- predict(modFit, testPC)

This leads to a 100% correct prediction on the course project submission.
