####This file contains the method comparison on the reduced balanced training and testing sets  ####
####(results found in table 2). Note that for the methods that contain some kind of random      ####
####component (NN, RF), the results may not match exactly what is in the paper, but they should ####
####be close. The other methods should match exactly what is seen in the paper (rounded to 4    ####
####decimal places). The commented code at the beginning can be used to recreate the reduced    ####
####data sets, or the sets can simply be read in.                                               ####

library(keras)
library(MASS)
library(microbenchmark)
library(randomForest)
library(kernlab)
library(png)
library(EBImage)

#create training data#
#set.seed(5118)
#tr.n<-51353
#back.n<-floor(0.60*tr.n)
#plant.n<-tr.n-back.n

#sample plant class#
#Y.plant.index<-which(Y==1)
#X.plant<-X[Y.plant.index,]
#Y.plant<-Y[Y.plant.index]
#plant.index<-sample(length(Y.plant.index),plant.n,replace=F)
#train.plant<-cbind(X.plant[plant.index,],Y.plant[plant.index])

#Y.plant.left<-Y.plant[-plant.index]
#X.plant.left<-X.plant[-plant.index,]

#sample background class#
#Y.back.index<-which(Y==0)
#X.back<-X[Y.back.index,]
#Y.back<-Y[Y.back.index]
#back.index<-sample(length(Y.back.index),back.n,replace=F)
#train.back<-cbind(X.back[back.index,],Y.back[back.index])

#train.dat<-rbind(train.plant,train.back)
#train.dat<-train.dat[sample(dim(train.dat)[1],dim(train.dat)[1],replace=F),]

#Y.back.left<-Y.back[-back.index]
#X.back.left<-X.back[-back.index,]


#create test data#
#ts.n<-10000
#back.n.ts<-floor(0.60*ts.n)
#plant.n.ts<-ts.n-back.n.ts

#plant.ts.index<-sample(length(Y.plant.left),plant.n.ts,replace=F)
#test.plant<-cbind(X.plant.left[plant.ts.index,],Y.plant.left[plant.ts.index])

#back.ts.index<-sample(length(Y.back.left),back.n.ts,replace=F)
#test.back<-cbind(X.back.left[back.ts.index,],Y.back.left[back.ts.index])

#test.dat<-rbind(test.plant,test.back)
#test.dat<-test.dat[sample(dim(test.dat)[1],dim(test.dat)[1],replace=F),]

#saveRDS(train.dat,"MethodComp/trd_bal.RDS")
#saveRDS(test.dat,"MethodComp/tsd_bal.RDS")
tr.nb<-readRDS("Data/BalMethodCompTr.RDS")
ts.nb<-readRDS("Data/BalMethodCompTs.RDS")

tr.rgb<-tr.nb[,c(5,14,23,28)]
ts.rgb<-ts.nb[,c(5,14,23,28)]

###########################################LDA###############################################
###no neighborhood information###
lda.rgb<-lda(x=tr.rgb[,1:3],grouping=tr.rgb[,4])

ptr.lda.rgb<-predict(lda.rgb,newdata = tr.rgb[,1:3])$class  #predict on training set
table(pred=ptr.lda.rgb,truth=tr.rgb[,4])                    #confusion matrix
prop.table(table(pred=ptr.lda.rgb,truth=tr.rgb[,4]))        #c. matrix proportions
length(which(ptr.lda.rgb!=tr.rgb[,4]))/dim(tr.rgb)[1]        #misclassification rate

pts.lda.rgb<-predict(lda.rgb,newdata = ts.rgb[,1:3])$class  #predict on test set
table(pred=pts.lda.rgb,truth=ts.rgb[,4])                    #confusion matrix
prop.table(table(pred=pts.lda.rgb,truth=ts.rgb[,4]))        #c. matrix proportions
length(which(pts.lda.rgb!=ts.rgb[,4]))/dim(ts.rgb)[1]        #misclassification rate

###neighborhood information###
lda.nb<-lda(x=tr.nb[,1:27],grouping=tr.nb[,28])

ptr.lda.nb<-predict(lda.nb,newdata = tr.nb[,1:27])$class  #predict on training set
table(pred=ptr.lda.nb,truth=tr.nb[,28])                    #confusion matrix
prop.table(table(pred=ptr.lda.nb,truth=tr.nb[,28]))        #c. matrix proportions
length(which(ptr.lda.nb!=tr.nb[,28]))/dim(tr.nb)[1]        #misclassification rate

pts.lda.nb<-predict(lda.nb,newdata = ts.nb[,1:27])$class  #predict on test set
table(pred=pts.lda.nb,truth=ts.nb[,28])                    #confusion matrix
prop.table(table(pred=pts.lda.nb,truth=ts.nb[,28]))        #c. matrix proportions
length(which(pts.lda.nb!=ts.nb[,28]))/dim(ts.nb)[1]        #misclassification rate

#########################################QDA###############################################
###no neighborhood information###
qda.rgb<-qda(x=tr.rgb[,1:3],grouping=tr.rgb[,4])

ptr.qda.rgb<-predict(qda.rgb,newdata = tr.rgb[,1:3])$class  #predict on training set
table(pred=ptr.qda.rgb,truth=tr.rgb[,4])                    #confusion matrix
prop.table(table(pred=ptr.qda.rgb,truth=tr.rgb[,4]))        #c. matrix proportions
length(which(ptr.qda.rgb!=tr.rgb[,4]))/dim(tr.rgb)[1]        #misclassification rate

pts.qda.rgb<-predict(qda.rgb,newdata = ts.rgb[,1:3])$class  #predict on test set
table(pred=pts.qda.rgb,truth=ts.rgb[,4])                    #confusion matrix
prop.table(table(pred=pts.qda.rgb,truth=ts.rgb[,4]))        #c. matrix proportions
length(which(pts.qda.rgb!=ts.rgb[,4]))/dim(ts.rgb)[1]        #misclassification rate

###neighborhood information###
qda.nb<-qda(x=tr.nb[,1:27],grouping=tr.nb[,28])

ptr.qda.nb<-predict(qda.nb,newdata = tr.nb[,1:27])$class  #predict on training set
table(pred=ptr.qda.nb,truth=tr.nb[,28])                    #confusion matrix
prop.table(table(pred=ptr.qda.nb,truth=tr.nb[,28]))        #c. matrix proportions
length(which(ptr.qda.nb!=tr.nb[,28]))/dim(tr.nb)[1]        #misclassification rate

pts.qda.nb<-predict(qda.nb,newdata = ts.nb[,1:27])$class  #predict on test set
table(pred=pts.qda.nb,truth=ts.nb[,28])                    #confusion matrix
prop.table(table(pred=pts.qda.nb,truth=ts.nb[,28]))        #c. matrix proportions
length(which(pts.qda.nb!=ts.nb[,28]))/dim(ts.nb)[1]        #misclassification rate

##########################################Random Forest######################################
###no neighborhood###
tr.rf<-as.data.frame(tr.rgb)
ts.rf<-as.data.frame(ts.rgb)

rf.rgb<-randomForest(x=tr.rf[,1:3],y=as.factor(tr.rf[,4]),ntree=500,classwt=c(0.9743,0.0257))
ptr.rf.rgb<-predict(rf.rgb,newdata=tr.rf[,1:3],type="class")
table(pred=ptr.rf.rgb,truth=tr.rgb[,4])                    #confusion matrix
prop.table(table(pred=ptr.rf.rgb,truth=tr.rgb[,4]))        #c. matrix proportions
length(which(ptr.rf.rgb!=tr.rf[,4]))/dim(tr.rgb)[1]        #misclassification rate

pts.rf.rgb<-predict(rf.rgb,newdata = ts.rf[,1:3],type="class")  #predict on test set
table(pred=pts.rf.rgb,truth=ts.rf[,4])                    #confusion matrix
prop.table(table(pred=pts.rf.rgb,truth=ts.rf[,4]))        #c. matrix proportions
length(which(pts.rf.rgb!=ts.rf[,4]))/dim(ts.rf)[1]        #misclassification rate

###neighborhood information###
trn.rf<-as.data.frame(tr.nb)
tsn.rf<-as.data.frame(ts.nb)

rf.nb<-randomForest(x=trn.rf[,1:27],y=as.factor(trn.rf[,28]),ntree=500,classwt=c(0.9743,0.0257))
ptr.rf.nb<-predict(rf.nb,newdata=trn.rf[,1:27],type="class")
table(pred=ptr.rf.nb,truth=trn.rf[,28])                    #confusion matrix
prop.table(table(pred=ptr.rf.nb,truth=trn.rf[,28]))        #c. matrix proportions
length(which(ptr.rf.nb!=trn.rf[,28]))/dim(trn.rf)[1]        #misclassification rate

pts.rf.nb<-predict(rf.nb,newdata = tsn.rf[,1:27],type="class")  #predict on test set
table(pred=pts.rf.nb,truth=tsn.rf[,28])                    #confusion matrix
prop.table(table(pred=pts.rf.nb,truth=tsn.rf[,28]))        #c. matrix proportions
length(which(pts.rf.nb!=tsn.rf[,28]))/dim(tsn.rf)[1]  

##############################################SVM############################################
###no neighborhood information###
svm.rgb<-ksvm(x=tr.rgb[,1:3],y=tr.rgb[,4],kernel="rbfdot",type="C-svc")

ptr.svm.rgb<-predict(svm.rgb,newdata = tr.rgb[,1:3])        #predict on training set
table(pred=ptr.svm.rgb,truth=tr.rgb[,4])                    #confusion matrix
prop.table(table(pred=ptr.svm.rgb,truth=tr.rgb[,4]))        #c. matrix proportions
length(which(ptr.svm.rgb!=tr.rgb[,4]))/dim(tr.rgb)[1]       #misclassification rate

pts.svm.rgb<-predict(svm.rgb,newdata = ts.rgb[,1:3])        #predict on test set
table(pred=pts.svm.rgb,truth=ts.rgb[,4])                    #confusion matrix
prop.table(table(pred=pts.svm.rgb,truth=ts.rgb[,4]))        #c. matrix proportions
length(which(pts.svm.rgb!=ts.rgb[,4]))/dim(ts.rgb)[1]        #misclassification rate

###neighborhood information###
svm.nb<-ksvm(x=tr.nb[,1:27],y=tr.nb[,28],kernel="rbfdot",type="C-svc")

ptr.svm.nb<-predict(svm.nb,newdata = tr.nb[,1:27])         #predict on training set
table(pred=ptr.svm.nb,truth=tr.nb[,28])                    #confusion matrix
prop.table(table(pred=ptr.svm.nb,truth=tr.nb[,28]))        #c. matrix proportions
length(which(ptr.svm.nb!=tr.nb[,28]))/dim(tr.nb)[1]        #misclassification rate

pts.svm.nb<-predict(svm.nb,newdata = ts.nb[,1:27])         #predict on test set
table(pred=pts.svm.nb,truth=ts.nb[,28])                    #confusion matrix
prop.table(table(pred=pts.svm.nb,truth=ts.nb[,28]))        #c. matrix proportions
length(which(pts.svm.nb!=ts.nb[,28]))/dim(ts.nb)[1]        #misclassification rate

##########################################Neural Network#####################################
###no neighborhood information###
X.rgb<-tr.rgb[,1:3]
Y.rgb<-tr.rgb[,4]
Y.lab.rgb<-to_categorical(Y.rgb,2)

mlp<-keras_model_sequential()
mlp %>%
  layer_dense(units=1024,input_shape=3,activation = "relu") %>%
  layer_dropout(0.45) %>%
  layer_dense(units=512,activation="relu") %>%
  layer_dropout(0.35) %>%
  layer_dense(units=1,activation="sigmoid")

summary(mlp)

mlp %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(0.001),
  metrics = c('accuracy')
)

history <- mlp %>% fit(
  X.rgb, Y.rgb, 
  epochs = 20, batch_size = 1024, 
  validation_split = 0.01
)

train.prob<-mlp %>% predict_proba(X.rgb)
ptr.nn.rgb<-rep(0,times=length(train.prob))
ptr.nn.rgb[train.prob>=0.95]<-1
table(pred=ptr.nn.rgb,truth=tr.rgb[,4])
prop.table(table(pred=ptr.nn.rgb,truth=tr.rgb[,4]))
length(which(ptr.nn.rgb!=tr.rgb[,4]))/dim(tr.rgb)[1]

test.prob<-mlp %>% predict_proba(ts.rgb[,1:3])
pts.nn.rgb<-rep(0,times=length(test.prob))
pts.nn.rgb[test.prob>=0.95]<-1
table(pred=pts.nn.rgb,truth=ts.rgb[,4])
prop.table(table(pred=pts.nn.rgb,truth=ts.rgb[,4]))
length(which(pts.nn.rgb!=ts.rgb[,4]))/dim(ts.rgb)[1]

###neighborhood information###
X.nb<-tr.nb[,1:27]
Y.nb<-tr.nb[,28]
Y.lab.nb<-to_categorical(Y.nb,2)

mlp.nb<-keras_model_sequential()
mlp.nb %>%
  layer_dense(units=1024,input_shape=27,activation = "relu") %>%
  layer_dropout(0.45) %>%
  layer_dense(units=512,activation="relu") %>%
  layer_dropout(0.35) %>%
  layer_dense(units=1,activation="sigmoid")

summary(mlp.nb)

mlp.nb %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(0.001),
  metrics = c('accuracy')
)

history <- mlp.nb %>% fit(
  X.nb, Y.nb, 
  epochs = 20, batch_size = 1024, 
  validation_split = 0.01
)

train.prob<-mlp.nb %>% predict_proba(X.nb)
ptr.nn.nb<-rep(0,times=length(train.prob))
ptr.nn.nb[train.prob>=0.5]<-1
table(pred=ptr.nn.nb,truth=tr.nb[,28])
prop.table(table(pred=ptr.nn.nb,truth=tr.nb[,28]))
length(which(ptr.nn.nb!=tr.nb[,28]))/dim(tr.nb)[1]

test.prob<-mlp.nb %>% predict_proba(ts.nb[,1:27])
pts.nn.nb<-rep(0,times=length(test.prob))
pts.nn.nb[test.prob>=0.5]<-1
table(pred=pts.nn.nb,truth=ts.nb[,28])
prop.table(table(pred=pts.nn.nb,truth=ts.nb[,28]))
length(which(pts.nn.nb!=ts.nb[,28]))/dim(ts.rgb)[1]

#################Double Criteria Thresholding#####################
###Function for DC thresholding on data matrix###
thresh.on.matrix<-function(mat,t1,t2,wts){
  temp1<-rowSums(mat)
  temp2<-mat%*%wts
  temp1<-1*(temp1>t1)
  temp2<-1*(temp2>t2)
  out<-temp1*temp2
  return(out)
}

###Parameter search on training data###
set.seed(6841)
t1.round1<-sample(1:128,size=1000,replace=T)/255
t2.round1<-sample(1:20,size=1000,replace=T)/100
mis.class.r1<-c()
for(i in 1:length(t1.round1)){
  p.thresh<-thresh.on.matrix(tr.rgb[,1:3],t1.round1[i],t2.round1[i],c(-1,2,-1))
  mis.class.r1[i]<-length(which(p.thresh!=tr.rgb[,4]))/dim(tr.rgb)[1]
}
top100.r1<-order(mis.class.r1,decreasing=F)[1:100]
cbind(mis.class.r1[top100.r1],t1.round1[top100.r1],t2.round1[top100.r1])
t1.round2<-runif(500,min=min(t1.round1[top100.r1]),max=max(t1.round1[top100.r1]))
t2.round2<-runif(500,min=min(t2.round1[top100.r1]),max=max(t2.round1[top100.r1]))
mis.class.r2<-c()
for(i in 1:length(t1.round2)){
  p.thresh<-thresh.on.matrix(tr.rgb[,1:3],t1.round2[i],t2.round2[i],c(-1,2,-1))
  mis.class.r2[i]<-length(which(p.thresh!=tr.rgb[,4]))/dim(tr.rgb)[1]
}
top50.r2<-order(mis.class.r2,decreasing=F)[1:50]
cbind(mis.class.r2[top50.r2],t1.round2[top50.r2],t2.round2[top50.r2])

###Use t1=0.1,t2=0.085 to obtain error rates###
ptr.thresh<-thresh.on.matrix(tr.rgb[,1:3],.1,0.085,c(-1,2,-1))
table(pred=ptr.thresh,truth=tr.rgb[,4])                    #confusion matrix
prop.table(table(pred=ptr.thresh,truth=tr.rgb[,4]))        #c. matrix proportions
length(which(ptr.thresh!=tr.rgb[,4]))/dim(tr.rgb)[1] 

pts.thresh<-thresh.on.matrix(ts.rgb[,1:3],0.1,0.085,c(-1,2,-1))
table(pred=pts.thresh,truth=ts.rgb[,4])                    #confusion matrix
prop.table(table(pred=pts.thresh,truth=ts.rgb[,4]))        #c. matrix proportions
length(which(pts.thresh!=ts.rgb[,4]))/dim(ts.rgb)[1]        #misclassification rate

#################Binary Threshold#####################
###function for binary threshold on data matrix###
bin.thresh<-function(mat,thresh){
  gray<-rowSums(mat)/3
  out<-1*(gray<thresh)
  return(out)
}

###parameter search on training data###
bt.parm<-c(0:255)/255
mis.class<-c()
for(i in 1:length(bt.parm)){
  p.binthresh<-bin.thresh(tr.rgb[,1:3],bt.parm[i])
  mis.class[i]<-length(which(p.binthresh!=tr.rgb[,4]))/dim(tr.rgb)[1]
}

top20.binthresh<-head(order(mis.class),20)
tr.res.binthresh<-cbind(mis.class[top20.binthresh],bt.parm[top20.binthresh])

###Use results of parameter serach (t=0.4) to obtain error rates###
ptr.thresh<-bin.thresh(tr.rgb[,1:3],0.4)
table(pred=ptr.thresh,truth=tr.rgb[,4])                    #confusion matrix
prop.table(table(pred=ptr.thresh,truth=tr.rgb[,4]))        #c. matrix proportions
length(which(ptr.thresh!=tr.rgb[,4]))/dim(tr.rgb)[1] 

pts.thresh<-bin.thresh(ts.rgb[,1:3],0.4)
table(pred=pts.thresh,truth=ts.rgb[,4])                    #confusion matrix
prop.table(table(pred=pts.thresh,truth=ts.rgb[,4]))        #c. matrix proportions
length(which(pts.thresh!=ts.rgb[,4]))/dim(ts.rgb)[1]   

