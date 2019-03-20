####This program extracts plant pixels for training data. The input image must first be cropped####
####to include only the plant and the white background (see, for example, CroppedImg1.png      ####
####and CroppedImg2.png). Programs from the file ImageConversion.R are needed to successfully  ####
####run this program.                                                                          ####

library(png)
library(EBImage)

cropped.img<-readPNG("Images/CroppedImg1.png")#Read-in image and convert to data matrix
cropped.use<-img.to.mat(cropped.img)

km.cropped<-kmeans(cropped.use,centers=3)     #Run K-means with 3 clusters and obtain predicted
pred.class<-km.cropped$cluster                #class for each pixel 

pred.plot<-rep(0,times=length(pred.class))    #Classify each pixel as 0 or 1 based on whether
plant.class<-1                                #it belongs to plant class or not. Plant class
pred.plot[pred.class==plant.class]<-1         #may need to be changed to 2 or 3, see below.

predicted.image<-matrix(NA,dim(cropped.img)[1],dim(cropped.img)[2])   #Create binary image
for(i in 1:dim(cropped.img)[1]){
  predicted.image[i,]<-pred.plot[((i-1)*dim(cropped.img)[2]+1):(i*dim(cropped.img)[2])]
}

writePNG(predicted.image,"Results/CroppedImg1Binary.png")   #Save resulting binary image as png

###Note that by the nature of the K-means algorithm, the plant class may not always be    ###
###labeled as the class 1. For this reason, it is important to view the output before     ###
###going further. If the resulting image does not look like Figure 5(d) from the paper,   ###
###change plant.class until it is (but do not rerun K-means, can restart at line 15).     ###
###We have not yet found a cropped image that fails to be well-segmented by this method,  ###
###but we always need to check to make sure we have the correct plant class.              ###

sp.data.with.labels<-cbind(cropped.use,pred.plot)   #combine features with labels
plant.examples<-sp.data.with.labels[sp.data.with.labels[,4]==1,] #extract plant pixels

###This process can now be repeated on as many cropped images as needed to produce enough ###
###training examples for the plant class. This can also be done similarly if we want to   ###
###include neighborhood features as well (see below).                                     ###

cropped.nb<-img.to.mat.nbhd(cropped.img)
nb.data.with.labels<-cbind(cropped.nb,pred.plot)   
plant.examples.nb<-nb.data.with.labels[nb.data.with.labels[,28]==1,] 
