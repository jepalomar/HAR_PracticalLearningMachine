library(caret);library(ggplot2);library(e1071)
setwd("C:\\Users\\jepal\\Documents\\Joan\\Coursera\\JohnHopkins_DataScience\\PracticalMachineLearning\\CourseProject")
training = read.csv("pml-training.csv", na.strings=c('#DIV/0!', '', 'NA','NaN') ,stringsAsFactors = F)
testing = read.csv("pml-testing.csv", na.strings=c('#DIV/0!', '', 'NA','NaN') ,stringsAsFactors = F)
head(training)
names(training)
# Some variables are irrelevant, such as user_name or timestamps. We will remove them
training<-training[,-c(1:7)]
# we treat classe as a factor
training$classe<-as.factor(training$classe)
# identify columns with small number of NA
goodCols<-colnames(training)[colSums(is.na(training))/length(training$roll_belt)<0.75]
myTrain<-training[,names(training) %in% goodCols]
# now we only keep the rows with no na in them
myTrain<-na.omit(myTrain) # it doesn't change the size

# do the same for testing
testing<-testing[,-c(1:7)]
myTest<-testing[,names(testing) %in% goodCols]
# we get only 52 columns instead of 53, because in training we get the classe column 
# (the one to be predicted), but in testing we have problem_id column, which wil not 
# be relevant in our prediction model.
myTest<-na.omit(myTest)

# define training and testing set from myTrain
inTrain<-createDataPartition(y=myTrain$classe,p=0.6,list=FALSE)
newTrain<-myTrain[inTrain,]
newTest<-myTrain[-inTrain,]

# train a random forest model
modRF<-train(classe~.,data=newTrain,method="rf",preProcess = c("zv","pca"))
rf<-predict(modRF,newdata=newTrain)
rf<-as.factor(rf)
confusionMatrix(rf,newTrain$classe)
table(rf,newTrain$classe)
# train a gbm model
modGBM<-train(classe~.,method="gbm",data=newTrain,verbose=FALSE,preProcess = c("zv","pca"))
gbm<-predict(modGBM,newdata=newTrain)
gbm<-as.factor(gbm)
confusionMatrix(gbm,newTrain$classe)
table(gbm,newTrain$classe)
# train a rpart model
modRPart<-train(classe~.,data=newTrain,method="rpart",preProcess = c("zv","pca"))
rpart<-predict(modRPart,newdata=newTrain)
rpart<-as.factor(rpart)
confusionMatrix(rpart,newTrain$classe)
table(rpart,newTrain$classe)

# we combine models now to see if we can get a better prediction
combinedTrain<-data.frame(rf,gbm,rpart,newTrain$classe)
names(combinedTrain)<-c("rf","gbm","rpart","classe")
finalMod<-train(classe~.,data=combinedTrain,method="rf")
myPred<-predict(finalMod,newdata=combinedTrain)
confusionMatrix(myPred,combinedTrain$classe)

### Let's see how the models work with the training test

# random forest
rf<-predict(modRF,newdata=newTest)
confusionMatrix(rf,newTest$classe)
table(rf,newTest$classe)

# gbm
gbm<-predict(modGBM,newdata=newTest)
confusionMatrix(gbm,newTest$classe)
# rpart
rpart<-predict(modRPart,newdata=newTest)
confusionMatrix(rmsePart,newTest$classe)

# combined model
rf<-predict(modRF,newdata=newTest)
gbm<-predict(modGBM,newdata=newTest)
rpart<-predict(modRPart,newdata=newTest)
x<-data.frame(rf,gbm,rpart,newTest$classe)
names(x)<-c("rf","gbm","rpart","classe")
newPred<-predict(finalMod,newdata=x)
confusionMatrix(newPred,x$classe)

classe<-predict(modRF,newdata=testing)
data.frame(testing$problem_id,classe)
