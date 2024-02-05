install.packages("readxl")
library("readxl")
library(e1071)
library(mnormt)
library (stats)
library(dplyr)
library(ggplot2)
library(ggfortify)

beans <- read_excel(path="Dry_Bean_Dataset.xlsx")
beandata <- data.frame(beans)

## Here I'm using Principle Component Analysis to reduce the dimensionality of the dataset
beansPCA <- prcomp(beandata[,-17],scale=T)
print(summary(beansPCA))
print(beansPCA)
#this gets the first two principle components since the summary tells us that they explain 82% of the data
pc1 <- beansPCA$rotation[,1]
print(pc1)
pc2 <- beansPCA$rotation[,2]
print(pc2)

##after applying PCA I was able to remove most attributs and keep 4 significant ones
idxs <- sample(1:nrow(beandata),as.integer(0.8*nrow(beandata)))
trainbeans <- beandata[idxs,c("AspectRation","EquivDiameter","Compactness","ShapeFactor3","Class")]
testbeans <- beandata[-idxs,c("AspectRation","EquivDiameter","Compactness","ShapeFactor3","Class")]

x1=trainbeans[,-5]
y1=trainbeans$Class
x2=testbeans[,-5]
y2=testbeans$Class
NBclass <- naiveBayes(x1,y1,data=trainbeans)
NB_pred <- predict(NBclass, testbeans, type="class")
t <- table(NB_pred, testbeans$Class,dnn=c("Prediction","Actual"))
print(t)
n1 <- sum(t[1,])
ns <- t[1,1]
Acc1=ns/n1
print(Acc1)
n2 <- sum(t[2,])
ns <- t[2,2]
print(ns/n2)
n3 <- sum(t[3,])
ns <- t[3,3]
print(ns/n3)
n4 <- sum(t[4,])
ns <- t[4,4]
print(ns/n4)
n5 <- sum(t[5,])
ns <- t[5,5]
print(ns/n5)
n6 <- sum(t[6,])
ns <- t[6,6]
print(ns/n6)
n7 <- sum(t[7,])
ns <- t[7,7]
print(ns/n7)
accuracy <- function(x){sum(diag(x))/(sum(rowSums(x)))*100}
accuracy(t)

##Naive Bayes with normal distribution and 2D Gaussian Distribution Plot
data1 <- trainbeans[which(trainbeans[,5]=="BARBUNYA"),]
data2 <- trainbeans[which(trainbeans[,5]=="BOMBAY"),]
data3 <- trainbeans[which(trainbeans[,5]=="CALI"),]
data4 <- trainbeans[which(trainbeans[,5]=="DERMASON"),]
data5 <- trainbeans[which(trainbeans[,5]=="HOROZ"),]
data6 <- trainbeans[which(trainbeans[,5]=="SEKER"),]
data7 <- trainbeans[which(trainbeans[,5]=="SIRA"),]

n1 <- nrow(data1)
n2 <- nrow(data2)
n3 <- nrow(data3)
n4 <- nrow(data4)
n5 <- nrow(data5)
n6 <- nrow(data6)
n7 <- nrow(data7)

ntotal=n1+n2+n3+n4+n5+n6+n7
pc1=n1/ntotal
pc2=n2/ntotal
pc3=n3/ntotal
pc4=n4/ntotal
pc5=n5/ntotal
pc6=n6/ntotal
pc7=n7/ntotal

data1nc <- data1[,-5]
data2nc <- data2[,-5]
data3nc <- data3[,-5]
data4nc <- data4[,-5]
data5nc <- data5[,-5]
data6nc <- data6[,-5]
data7nc <- data7[,-5]

me11 <- mean(data1nc$EquivDiameter)
me12 <- mean(data1nc$AspectRation)

me21 <- mean(data2nc$EquivDiameter)
me22 <- mean(data2nc$AspectRation)

me31 <- mean(data3nc$EquivDiameter)
me32 <- mean(data3nc$AspectRation)

me41 <- mean(data4nc$EquivDiameter)
me42 <- mean(data4nc$AspectRation)

me51 <- mean(data5nc$EquivDiameter)
me52 <- mean(data5nc$AspectRation)

me61 <- mean(data6nc$EquivDiameter)
me62 <- mean(data6nc$AspectRation)

me71 <- mean(data7nc$EquivDiameter)
me72 <- mean(data7nc$AspectRation)

mu1 <- c(me11,me12)
mu2 <- c(me21,me22)
mu3 <- c(me31,me32)
mu4 <- c(me41,me42)
mu5 <- c(me51,me52)
mu6 <- c(me61,me62)
mu7 <- c(me71,me72)

d1 <- data.frame(data1nc$EquivDiameter,data1nc$AspectRation)
cov1 <- cov(d1)

d2 <- data.frame(data2nc$EquivDiameter,data2nc$AspectRation)
cov2 <- cov(d2)

d3 <- data.frame(data3nc$EquivDiameter,data3nc$AspectRation)
cov3 <- cov(d3)

d4 <- data.frame(data4nc$EquivDiameter,data4nc$AspectRation)
cov4 <- cov(d4)

d5 <- data.frame(data5nc$EquivDiameter,data5nc$AspectRation)
cov5 <- cov(d5)

d6 <- data.frame(data6nc$EquivDiameter,data6nc$AspectRation)
cov6 <- cov(d6)

d7 <- data.frame(data7nc$EquivDiameter,data7nc$AspectRation)
cov7 <- cov(d7)

#naive bayes classification
p1 <- numeric()
p2 <- numeric()
p3 <- numeric()
p4 <- numeric()
p5 <- numeric()
p6 <- numeric()
p7 <- numeric()

cl <- integer()

for (i in 1:nrow(testbeans)){
  x <- testbeans$EquivDiameter[i]
  y <- testbeans$AspectRation[i]
  f1=dmnorm(c(x,y),mu1,cov1)
  f2=dmnorm(c(x,y),mu2,cov2)
  f3=dmnorm(c(x,y),mu3,cov3)
  f4=dmnorm(c(x,y),mu4,cov4)
  f5=dmnorm(c(x,y),mu5,cov5)
  f6=dmnorm(c(x,y),mu6,cov6)
  f7=dmnorm(c(x,y),mu7,cov7)
  
  p1[i]=pc1*(f1/(f1+f2+f3+f4+f5+f6+f7))
  p2[i]=pc2*(f2/(f1+f2+f3+f4+f5+f6+f7))
  p3[i]=pc3*(f3/(f1+f2+f3+f4+f5+f6+f7))
  p4[i]=pc4*(f4/(f1+f2+f3+f4+f5+f6+f7))
  p5[i]=pc5*(f5/(f1+f2+f3+f4+f5+f6+f7))
  p6[i]=pc6*(f6/(f1+f2+f3+f4+f5+f6+f7))
  p7[i]=pc7*(f7/(f1+f2+f3+f4+f5+f6+f7))
  
  proball <- cbind(p1[i],p2[i],p3[i],p4[i],p5[i],p6[i],p7[i])
  nm <- which(proball==max(proball))
  cl[i]=nm
}

#table
#confusion matrix

t <- table(cl, testbeans$Class,dnn=c("Prediction","Actual"))
t
n1 <- sum(t[1,])
ns <- t[1,1]
Acc1=ns/n1
print(Acc1)
n2 <- sum(t[2,])
ns <- t[2,2]
print(ns/n2)
n3 <- sum(t[3,])
ns <- t[3,3]
print(ns/n3)
n4 <- sum(t[4,])
ns <- t[4,4]
print(ns/n4)
n5 <- sum(t[5,])
ns <- t[5,5]
print(ns/n5)
n6 <- sum(t[6,])
ns <- t[6,6]
print(ns/n6)
n7 <- sum(t[7,])
ns <- t[7,7]
print(ns/n7)
accuracy <- function(x){sum(diag(x))/(sum(rowSums(x)))*100}
accuracy(t)

#plot contours of 2D Gaussian distribution 
x.points <- seq(160,560,length.out=200)
y.points <- seq(1,2.5,length.out=200)
z1 <- matrix(0.0,nrow=200,ncol=200)
z2 <- matrix(0.0,nrow=200,ncol=200)
z3 <- matrix(0.0,nrow=200,ncol=200)
z4 <- matrix(0.0,nrow=200,ncol=200)
z5 <- matrix(0.0,nrow=200,ncol=200)
z6 <- matrix(0.0,nrow=200,ncol=200)
z7 <- matrix(0.0,nrow=200,ncol=200)

for (i in 1:200){
  for (j in 1:200){
    z1[i,j] <- dmnorm(c(x.points[i],y.points[j]),
                      mu1,cov1)
    z2[i,j] <- dmnorm(c(x.points[i],y.points[j]),
                      mu2,cov2)
    z3[i,j] <- dmnorm(c(x.points[i],y.points[j]),
                      mu3,cov3)
    z4[i,j] <- dmnorm(c(x.points[i],y.points[j]),
                      mu4,cov4)
    z5[i,j] <- dmnorm(c(x.points[i],y.points[j]),
                      mu5,cov5)
    z6[i,j] <- dmnorm(c(x.points[i],y.points[j]),
                      mu6,cov6)
    z7[i,j] <- dmnorm(c(x.points[i],y.points[j]),
                      mu7,cov7)
  }
}
contour(x.points,y.points,z1,col = "red")
contour(x.points,y.points,z2,add=T,col = "blue")
contour(x.points,y.points,z3,add=T,col = "purple")
contour(x.points,y.points,z4,add=T,col = "black")
contour(x.points,y.points,z5,add=T,col = "green")
contour(x.points,y.points,z6,add=T,col = "cyan")
contour(x.points,y.points,z7,add=T,col = "brown")
title(main="2D Gaussian Distribution",xlab = "EquivDiameter",ylab = "AspectRation")
legend("topright",legend = c("Barbunya","Bombay","Cali","Dermason","Horoz","Seker","Sira"),cex =.45,lwd = 4,col = c("red","blue","purple","black","green","cyan","brown"))
dev.off()

##SVM Classification
idxs <- sample(1:nrow(trainbeans),as.integer(0.2*nrow(trainbeans)))
databeans <- trainbeans[idxs,c("EquivDiameter","ShapeFactor3","Class")]
svmtune <- tune.svm(as.factor(Class)~.,data=databeans,
                    type="C-classification",
                    control=tune.control(sampling=c("cross"),
                                         cross=5,best.model=T,performances=T),
                    kernel="linear",cost=2^(-5:5))

bestC <- svmtune$best.parameters[[1]]
svmm <- svm(as.factor(Class)~.,data=databeans,
            type="C-classification",kernal="linear",
            cost=bestC,scale=T,cross=5)


predSVM <- predict(svmm,databeans,decision.value=T)
t <- table(databeans$Class,predSVM)

print(t)
n1 <- sum(t[1,])
ns <- t[1,1]
Acc1=ns/n1
print(Acc1)
n2 <- sum(t[2,])
ns <- t[2,2]
print(ns/n2)
n3 <- sum(t[3,])
ns <- t[3,3]
print(ns/n3)
n4 <- sum(t[4,])
ns <- t[4,4]
print(ns/n4)
n5 <- sum(t[5,])
ns <- t[5,5]
print(ns/n5)
n6 <- sum(t[6,])
ns <- t[6,6]
print(ns/n6)
n7 <- sum(t[7,])
ns <- t[7,7]
print(ns/n7)
accuracy <- function(x){sum(diag(x))/(sum(rowSums(x)))*100}
accuracy(t)
plot(svmm,data=databeans,color.palette=terrain.colors)
dev.off()


##KMeans Clustering
databeans <- trainbeans[,c("AspectRation","EquivDiameter","Compactness","ShapeFactor3")]
wssplot <- function(data, nc=15, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within Groups Sum of Squares")}

# find the optimal number of clusters
wssplot(databeans)

# K-Means Cluster
km <-  kmeans(databeans,3)

# Cluster Plot
autoplot(km,databeans,frame=TRUE)

# Cluster Centers
km$centers


##KNN 
ran <- sample(1:nrow(beandata),0.9*nrow(beandata))
print(ran)
##the normalization function is created 
nor <- function(x) {(x-min(x))/(max(x)-min(x))}

##test function nor2
nor2 <- function(x){x-mean(x)/(sd(x))}

##normalization function is created
bean_nor <- as.data.frame(lapply(beandata[,c("AspectRation","EquivDiameter","Compactness","ShapeFactor3")], nor2))

bean_train <- bean_nor[ran,]
bean_test <- bean_nor[-ran,]

## also convert ordered factor to normal factor
bean_target <- as.factor(beandata[ran,"Class"])

##also convert ordered factor to normal factor 
test_target <- as.factor(beandata[-ran,"Class"])

##run knn function
library(class)
pr <- knn(bean_train,bean_test,cl=bean_target,k=10)

##create the confuction matrix 
t <- table(pr,test_target)

n1 <- sum(t[1,])
ns <- t[1,1]
Acc1=ns/n1
print(Acc1)
n2 <- sum(t[2,])
ns <- t[2,2]
print(ns/n2)
n3 <- sum(t[3,])
ns <- t[3,3]
print(ns/n3)
n4 <- sum(t[4,])
ns <- t[4,4]
print(ns/n4)
n5 <- sum(t[5,])
ns <- t[5,5]
print(ns/n5)
n6 <- sum(t[6,])
ns <- t[6,6]
print(ns/n6)
n7 <- sum(t[7,])
ns <- t[7,7]
print(ns/n7)

##check the accuracy
accuracy <- function(x){sum(diag(x))/(sum(rowSums(x)))*100}
accuracy(t)

##Decision Tree
library(rpart)

cfit <- rpart(as.factor(Class)~.,data=trainbeans,method="class")
name1="Decision Tree for Bean Data"
num=1
ext=".pdf"
name2=paste(name1,num,ext,sep='')
pdf(name2)
plot(cfit,uniform=T,main="Decision Tree for Bean Data")
text(cfit,use.n=T,all=T,cex=.5)
dev.off()
pred <- predict(cfit,data=trainbeans,type="class")
t <- table(trainbeans$Class,pred)
print(t)
d <- diag(t)
n <- sum(t)
overall_ac=d/n
n1 <- sum(t[1,])
ns <- t[1,1]
Acc1=ns/n1
print(Acc1)
n2 <- sum(t[2,])
ns <- t[2,2]
print(ns/n2)
n3 <- sum(t[3,])
ns <- t[3,3]
print(ns/n3)
n4 <- sum(t[4,])
ns <- t[4,4]
print(ns/n4)
n5 <- sum(t[5,])
ns <- t[5,5]
print(ns/n5)
n6 <- sum(t[6,])
ns <- t[6,6]
print(ns/n6)
n7 <- sum(t[7,])
ns <- t[7,7]
print(ns/n7)
accuracy(t)

##logistic regression
split <- sample.split(beandata, SplitRatio = 0.8)
split

train_reg <- subset(beandata, split == "TRUE")
test_reg <- subset(beandata, split == "FALSE")
# Training model

logistic_model <- glm(as.numeric(factor(Class)) ~ EquivDiameter + AspectRation + Compactness + ShapeFactor3,
                      data = train_reg, family = poisson)
logistic_model
plot(logistic_model)


# Summary
summary(logistic_model)

predict_reg <- predict(logistic_model,
                       test_reg, type = "response")
predict_reg

predict_reg <- round(predict_reg,digits = 0)
predict_reg

# Evaluating model accuracy
# using confusion matrix
t <- table(as.numeric(factor(test_reg$Class)), predict_reg)
t
n1 <- sum(t[1,])
ns <- t[1,1]
Acc1=ns/n1
print(Acc1)
n2 <- sum(t[2,])
ns <- t[2,2]
print(ns/n2)
n3 <- sum(t[3,])
ns <- t[3,3]
print(ns/n3)
n4 <- sum(t[4,])
ns <- t[4,4]
print(ns/n4)
n5 <- sum(t[5,])
ns <- t[5,5]
print(ns/n5)
n6 <- sum(t[6,])
ns <- t[6,6]
print(ns/n6)
n7 <- sum(t[7,])
ns <- t[7,7]
print(ns/n7)
accuracy(t)

