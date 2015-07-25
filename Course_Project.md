# Practical Machine Learning Course Project

## Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement, a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is to build a machine learning algorithm to predict activity quality from activity monitors and apply that algorithm to 20 test cases.


```r
for (package in c("caret", "doMC", "knitr", "xtable")) {
    if (!(require(package, character.only = TRUE, quietly = TRUE))) {
        install.packages(package)
        library(package, character.only = TRUE)
    }
}

options(xtable.comment = FALSE)
options(scipen = 5)
```



## Getting and reading the data

The data will be downloaded and saved in the working directory and then it will be read into R. The data is divided into a training dataset and a testing dataset. The first set will be used to build the machine learning algorithm and the second contains the 20 test cases that need to be predicted for the assignment.


```r
# Downloading data
urlTraining <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTesting <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
fileTraining <- "pml-training.csv"
fileTesting <- "pml-testing.csv"
if (!file.exists(fileTraining)) {
    download.file(urlTraining, fileTraining, method = "curl")
}
if (!file.exists(fileTesting)) {
    download.file(urlTesting, fileTesting, method = "curl")
}

# Reading data
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

# Quick look at the data
dim(training)
dim(testing)
head(training)[1:10]
```

```
## [1] 19622   160
## [1]  20 160
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt
## 1         no         11      1.41       8.07    -94.4
## 2         no         11      1.41       8.07    -94.4
## 3         no         11      1.42       8.07    -94.4
## 4         no         12      1.48       8.05    -94.4
## 5         no         12      1.48       8.07    -94.4
## 6         no         12      1.45       8.06    -94.4
```

Both datasets have 160 variables. In order to build a good machine learning algorithm, the number of variables (features) will be reduced.

## Data cleaning and data partition

The variables that should be eliminated are those with many missing values and those with zero or near zero variability, because they do not add anything to the model. The first variable will also be discarded because it contains only the observation number.


```r
# Remove variables where more than half its observations are missing values
index <- which(colMeans(is.na(training)) > 0.5)
subTraining <- training[,-index]

# Remove variables with near zero variance and first column, which is just the row number
index2 <- nearZeroVar(subTraining)
subdata <- subTraining[,-c(1,index2)]
dim(subdata)
```

```
## [1] 19622    58
```

By removing the initial column (1 variable), the variables with more than half its values being NAs (67 variables) and the variables with near zero variation (34 variables), the number of features was reduced from 160 to 58.

Next, the cleaned training set is divided into `trainData` (60% of the observations) that will be used to build the algorithm and `validationData` (40%) that will be used to estimate the out of sample error.


```r
set.seed(123)
indexPart <- createDataPartition(y = subdata$classe, p = 0.6, list = FALSE)
trainData <- subdata[indexPart,]
validationData <- subdata[-indexPart,]
```



## Fitting the models

Different models will be fitted and then one of them will be chosen based on accuracy. Since these computations can be time-consuming, particularly for random forest and boosting, parallelization will be used to decrease the time it takes to build the model. For each model, 10-fold cross-validation will be done, except for random forest, in which case 5-fold cross-validation will be done in order to decrease the time taken to build the model.


```r
# Set up parallelization
nCores <- detectCores()
registerDoMC(cores = nCores)

# Fitting models
modelRF <- train(classe ~ ., data = trainData, trControl = trainControl(method = "cv", number = 5),
                 method = "rf") # Random Forest
modelBoost <- train(classe ~ ., data = trainData, trControl = trainControl(method = "cv", number = 10),
                 method = "gbm") # Boosting
modelTrees <- train(classe ~ ., data = trainData, trControl = trainControl(method = "cv", number = 10),
                    method = "rpart") # Classification trees
modelLDA <- train(classe ~ ., data = trainData, trControl = trainControl(method = "cv", number = 10),
                    method = "lda") # Linear Discriminant Analysis
```


```r
accRF <- modelRF$results[2,2]
accBoost <- modelBoost$results[9,5]
accTrees <- modelTrees$results[1,2]
accLDA <- as.numeric(modelLDA$results[2])
values <- c(accRF, accBoost, accTrees, accLDA)
rows <- c("Random Forest", "Boosting", "Classification Trees", "Linear Discriminat Analysis")
tab <- data.frame(rows, round(values, 5))
names(tab) <- c("Model", "Accuracy")

print(xtable(tab, digits = c(0,0,5), align = c("l", "l", "r")), type = "html", include.rownames = FALSE,
      html.table.attributes = "align = 'center'")
```

<table align = 'center'>
<tr> <th> Model </th> <th> Accuracy </th>  </tr>
  <tr> <td> Random Forest </td> <td align="right"> 0.99839 </td> </tr>
  <tr> <td> Boosting </td> <td align="right"> 0.99601 </td> </tr>
  <tr> <td> Classification Trees </td> <td align="right"> 0.58983 </td> </tr>
  <tr> <td> Linear Discriminat Analysis </td> <td align="right"> 0.85462 </td> </tr>
   </table>

<br>

The greatest accuracy was obtained with the random forest model. Thus, this model will be used for predictions. As for the other models, boosting had accuracy almost equal to that of random forest and it could be used as well. Linear discriminant analysis performed well, but it is a sub-optimal choice. Classification trees performed poorly.


## Validation and out of sample error

Before proceeding with the predictions for the test cases, it is important to calculate the performance of the model with a different dataset. To do this, the validationData will be used to calculate the confusion matrix and the out of sample error for the model.


```r
# Confusion matrix
confusionMatrix(validationData$classe, predict(modelRF, newdata = validationData))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1518    0    0    0
##          C    0    2 1365    1    0
##          D    0    0    0 1286    0
##          E    0    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9995          
##                  95% CI : (0.9987, 0.9999)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9994          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9987   1.0000   0.9984   1.0000
## Specificity            1.0000   1.0000   0.9995   1.0000   0.9998
## Pos Pred Value         1.0000   1.0000   0.9978   1.0000   0.9993
## Neg Pred Value         1.0000   0.9997   1.0000   0.9997   1.0000
## Prevalence             0.2845   0.1937   0.1740   0.1642   0.1837
## Detection Rate         0.2845   0.1935   0.1740   0.1639   0.1837
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   0.9993   0.9998   0.9992   0.9999
```

The model had a high accuracy of 0.99949 or 99.949%.


```r
# Out of sample error
outSamErr <- 1- as.numeric(confusionMatrix(validationData$classe, predict(modelRF, newdata = validationData))$overall[1])
```

Therefore, the out of sample error for the model is 0.00051 or 0.051%.


## Prediction of test cases

Based on the random forest model, the activities predicted for the 20 test cases in the testing dataset are predicted to be:


```r
predict(modelRF, newdata = testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



## Additional Information

### Where to get the data?
All the data used in this project was obtained from the Weight Lifting Exercises Dataset, part of the [Human Activity Recognition] [1] project (1). The data can be downloaded at:

- [Training dataset] [2]

- [Testing dataset] [3]

[1]: http://groupware.les.inf.puc-rio.br/har "Human Activity Recognition website"

[2]: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv "Training dataset"

[3]: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv "Testing dataset"



### Bibliography

1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.



### R Session Information


```r
sessionInfo()
```

```
## R version 3.2.0 (2015-04-16)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: OS X 10.10.4 (Yosemite)
## 
## locale:
## [1] C
## 
## attached base packages:
## [1] splines   parallel  stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] MASS_7.3-40         rpart_4.1-9         plyr_1.8.2         
##  [4] gbm_2.1.1           survival_2.38-1     randomForest_4.6-10
##  [7] xtable_1.7-4        knitr_1.10.5        doMC_1.3.3         
## [10] iterators_1.0.7     foreach_1.4.2       caret_6.0-47       
## [13] ggplot2_1.0.1       lattice_0.20-31    
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.11.6         compiler_3.2.0      formatR_1.2        
##  [4] nloptr_1.0.4        class_7.3-12        tools_3.2.0        
##  [7] digest_0.6.8        lme4_1.1-7          evaluate_0.7       
## [10] nlme_3.1-120        gtable_0.1.2        mgcv_1.8-6         
## [13] Matrix_1.2-1        yaml_2.1.13         brglm_0.5-9        
## [16] SparseM_1.6         proto_0.3-10        e1071_1.6-4        
## [19] BradleyTerry2_1.0-6 stringr_1.0.0       gtools_3.5.0       
## [22] grid_3.2.0          nnet_7.3-9          rmarkdown_0.7      
## [25] minqa_1.2.4         reshape2_1.4.1      car_2.0-25         
## [28] magrittr_1.5        scales_0.2.4        codetools_0.2-11   
## [31] htmltools_0.2.6     pbkrtest_0.4-2      colorspace_1.2-6   
## [34] quantreg_5.11       stringi_0.4-1       munsell_0.4.2
```

