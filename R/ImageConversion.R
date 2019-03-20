####These programs allow for the conversion of raw images (in PNG format) into matrices that####
####can be used as the features for training supervised learing methods or for prediction.  ####
####The img.to.mat function simply gives the red, green, and blue intensities for each      ####
####pixel. So there are 3 columns in the resulting matrix, and as many rows as there are    ####
####pixels in the image. The img.to.mat.nbhd function includes the pixel intensities of the ####
####eight pixels surrounding each pixel of interest. So the resulting matrix contains 27    ####
####columns and as many rows as there are pixels in the image. Two packages are required to ####
####run these programs: png and EBImage (the code for installing the latter is included as  ####
####a comment, while standard installation procedure will suffice for the former).          ####

#source("http://bioconductor.org/biocLite.R")
#biocLite("EBImage")
library(EBImage)
library(png)

img.to.mat<-function(img){
  k<-1
  img.mat<-matrix(NA,nrow=dim(img)[1]*dim(img)[2],ncol=3)
  for(i in 1:dim(img)[1]){
    for(j in 1:dim(img)[2]){
      
      img.mat[k,]<-c(img[i,j,1],img[i,j,2],img[i,j,3])
      k<-k+1
    }
  }
  return(img.mat)
}

img.to.mat.nbhd<-function(img){
  k<-1
  img.pad<-array(NA,dim=c(dim(img)[1]+2,dim(img)[2]+2,3))
  img.pad[,,1]<-cbind(0,rbind(0,img[,,1],0),0)
  img.pad[,,2]<-cbind(0,rbind(0,img[,,2],0),0)
  img.pad[,,3]<-cbind(0,rbind(0,img[,,3],0),0)
  img.mat<-matrix(NA,nrow=(dim(img)[1])*(dim(img)[2]),ncol=27)
  for(i in 2:(dim(img.pad)[1]-1)){
    for(j in 2:(dim(img.pad)[2]-1)){
      img.mat[k,]<-c(as.vector(img.pad[((i-1):(i+1)),((j-1):(j+1)),1]),
                     as.vector(img.pad[((i-1):(i+1)),((j-1):(j+1)),2]),
                     as.vector(img.pad[((i-1):(i+1)),((j-1):(j+1)),3]))
      k<-k+1
    }
  }
  return(img.mat)
}

#Example using ExImg1.png#
ex.img<-readPNG("Images/ExImg1.png") #Read image in as array (using function from png package)
img.resize<-resize(ex.img,512,512)   #Resize image using EBImage function (not necessary)
ex.sp<-img.to.mat(img.resize)        #Convert image to (single pixel) data matrix
ex.nb<-img.to.mat.nbhd(img.resize)   #Convert image to (neighborhood) data matrix

#examine data sets#
head(ex.sp)
dim(ex.sp)

head(ex.nb)
dim(ex.nb)

