####This program uses the neural network described in the paper to turn raw images into binary ####
####images. Here, we first resize the raw image to 384x384 pixels, but this is not necessary.  ####
####Note, however, that larger images will take longer to construct binary images.             ####

mlp<-load_model_hdf5("Results/MLP6.hdf5") #read in model (can also use NeuralNetworkTraining.R
                                          #to train the model on your own machine)

img.test<-resize(readPNG("Images/ExImg1.png"),384,384)  #read in and resize image
test.mat<-img.to.mat.nbhd(img.test)                     #convert to data matrix (NB format)

test.prob<-mlp %>% predict_proba(test.mat) #use NN to obtain estimated probability that each
test.lab<-rep(0,times=length(test.prob))   #pixel belongs to the plant class. If the estimated
test.lab[test.prob>=0.95]<-1               #probability is higher than .95, it is classified as
                                           #a plant pixel

predicted.image<-matrix(NA,dim(img.test)[1],dim(img.test)[2])
for(i in 1:dim(predicted.image)[1]){
  predicted.image[i,]<-test.lab[((i-1)*dim(predicted.image)[2]+1):(i*dim(predicted.image)[2])]
}                                                    #convert predictions to binary image

writePNG(predicted.image,"Results/ExImg1Binary.png") #write binary image to a png
