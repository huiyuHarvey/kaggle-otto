rm(list = ls())

library('class')

raw_data <- read.csv("train.csv")

#X is the data frame with all observations and 93 features
#extracting from raw_data, 3 here is used in windows, but could be 2 in IOS.
X <- raw_data[, 3:length(raw_data)-1]
#Y is the data frame with all observations but only target column
Y <- raw_data[, length(raw_data)]

num_col <- ncol(X)
num_row <- nrow(X)

set.seed(1)

#sample the X and Y
random_row <- sample(num_row)
X <- X[random_row, ]
Y <- Y[random_row]

featureNum <- 32

#this is to do the PCA analysis
feature_PCA <- prcomp(X, center = TRUE, scale. = TRUE)

#the next 2 lines are to compute the PC value of the training data set
X_PCA <- predict(feature_PCA, X)
X_PCA <- as.data.frame(X_PCA)

#this is to subset the certain number of columns, which is the predict the better feature number
X_PCA_sub <- X_PCA[, 1:featureNum]

###set parameters for cross-validation
index <- c(1: num_row)
#capital K for KNN
K <- 50
#small k for k-fold
k <- 5
fold_size <- floor(num_row/k)
label_size <- 9
Y_hat_ts <- numeric(fold_size)
balanced_error_rates <- NULL
misclassification_rates <- NULL
logloss <- NULL


for (fold_index in 1:k){
    
    #ts_index is the index of observations of test set
    ts_index <- (((fold_index - 1) * fold_size + 1):(fold_index * fold_size))
    #tr_index is the index of observations of training set
    tr_index <- setdiff(index, ts_index)
    
    #tr_X is the training set with 93 features
    tr_X <- X_PCA_sub[tr_index, ]
    #tr_Y is the training set with target
    tr_Y <- Y[tr_index]
    #ts_X is the test set with 93 features
    ts_X <- X_PCA_sub[ts_index, ]
    #ts_Y is the test set with target
    ts_Y <- Y[ts_index]
	

    ###Predict using knn from class package 
    result <- knn(tr_X, ts_X, tr_Y, K)  

    ###compute the misclassification rate
	  #the fold_misclassification_rate is the mean misclassification rate of each fold
    fold_misclassification_rate <- mean(as.numeric(ts_Y != result))
	  print(paste(" Misclassification rate per fold: ", fold_misclassification_rate))
    misclassification_rates <- c(misclassification_rates, fold_misclassification_rate)
	
    ###compute balanced error rate
    confusion_matrix <- matrix(0, nrow = label_size, ncol = label_size)
    #get the confusion matrix
    for(i in 1 : length(ts_index)){
        real_target <- ts_Y[i]
        predicted_target <- result[i]
        confusion_matrix[predicted_target, real_target] = confusion_matrix[predicted_target, real_target] + 1
    }
	
    fold_BER <- 0
    for(label in 1 : label_size) {
        TP <- confusion_matrix[label, label]
        TP_FN <- sum(confusion_matrix[,label])
        fold_BER <- fold_BER + TP/TP_FN
    }
	
    fold_BER <- fold_BER / label_size
	  print(paste(" balanced error rate per fold: ", fold_BER))
    balanced_error_rates <- c(balanced_error_rates, fold_BER)
    
    
    
    #this is to change the format into submission format,
    #so that the result from KNN could be used to calculate logloss
    Y_hat_ts <- class.ind(result)
    ###compute the logloss
	  fold_logloss <- 0
	  for(i in length(ts_index)) {
		  
      sum_column <- sum(Y_hat_ts[i, ])
  		p_i <- Y_hat_ts[i, ]/sum_column
  		real_target <- ts_Y[i]
  		p_ij <- max(min(p_i[real_target], 1-10^-15), 10^-15)
  		log_p_ij <- log(p_ij)
  		fold_logloss <- fold_logloss + log_p_ij
      
	  }
	  fold_logloss <- -fold_logloss/length(ts_index)
	  print(paste(" logloss per fold: ", fold_logloss))
	  logloss <- c(logloss, fold_logloss)
    
}
final_misclassification_rate <- mean(misclassification_rates)
final_balanced_error_rate <- mean(balanced_error_rates)
final_logloss <- mean(logloss)

print("=======Cross-validation finish=========")
print(paste("Mis-Classification error rate: ", final_misclassification_rate))
print(paste("balanced error rate: ", final_balanced_error_rate))
print(paste("logloss: ", final_logloss))

error_rate <- c(final_misclassification_rate, final_balanced_error_rate, final_logloss)

bp <- barplot(error_rate, space = 0.01, width = 0.1,
              ylim = c(0,4),
              main = "K-Nearest-Neighbours evaluation", 
              xlab = "evaluation parameter",
              beside = TRUE,
              names.arg = c("Misclassification Rate", "Balanced Error Rate", "Logloss"),
              col = rainbow(3),
              cex.lab = 1.5,
              axes = TRUE,
)
text(bp, 0,round(error_rate, 4), cex = 1, pos = 3)
legend("topleft", c("Misclassification Rate", "Balanced Error Rate", "Logloss"),
       bty = "n", fill = rainbow(3), cex = 0.6)