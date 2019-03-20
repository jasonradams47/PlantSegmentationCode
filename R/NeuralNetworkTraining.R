####This program contains code needed for training the neural network on neighborhood data as ####
####described in the paper. The keras package is needed, and this package also requires an    ####
####installation of python on your machine.                                                   ####

library(keras)
X<-readRDS("Data/NbhdFeatures.rds")  #Read in data - obtained as described in paper
Y<-readRDS("Data/NbhdLabels.rds")
Y.lab<-to_categorical(Y,2)

#####Define and Train Network#####
mlp<-keras_model_sequential()
mlp %>%
  layer_dense(units=1024,input_shape=27,activation = "relu") %>%
  layer_dropout(0.45) %>%
  layer_dense(units=512,activation="relu") %>%
  layer_dropout(0.35) %>%
  layer_dense(units=1,activation="sigmoid")

#summary(mlp)  #can use to view a description of the network before training

mlp %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(0.001),
  metrics = c('accuracy')
)                                      #define optimization algorithm and loss function

history <- mlp %>% fit(
  X, Y, 
  epochs = 20, batch_size = 1024, 
  validation_split = 0.01
)                                      #train network (takes about 40 minutes on my machine)


###If you want to use the model trained here for image prediction, then you can save it and ###
###read it into the ImageBinarization.R file instead of using the one trained and used for  ###
###results in the paper. Since the data and the network are exactly the same, the results   ###
###should be similar (but because of the randomness inherent in training the neural network,###
###they will most likely not be exactly the same).                                          ###

save_model_hdf5(mlp,"Results/NN.hdf5") #saves the model trained here for future use

