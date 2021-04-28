# Machine learning to classify images

library(keras)

# Setup

# list of animals to model
animal_list <- c("Butterfly", "Cow", "Elephant", "Spider")

# number of output classes (i.e. fruits)
output_n <- length(animal_list)

# image size to scale down to (original images vary but about 600 x 800 px)
img_width <- 250
img_height <- 250
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
setwd("~/Documents/Uni/BIO8068/BIO8068_machine_learning/animals")

train_image_files_path <- "Training\\"
valid_image_files_path <- "Validation\\"

# Check number of images
length(list.files(train_image_files_path, recursive = TRUE))
length(list.files(valid_image_files_path, recursive = TRUE))

# Data generators and augmentation

# Rescale from 255 to between zero and 1
train_data_gen = image_data_generator(
  rescale = 1/255
)

valid_data_gen <- image_data_generator(
  rescale = 1/255
)  

# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = animal_list,
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = animal_list,
                                                    seed = 42)

# Check that things seem to have been read in OK
cat("Number of images per class:")

table(factor(train_image_array_gen$classes))

cat("Class labels vs index mapping")

train_image_array_gen$class_indices

# Final setup

# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32 # Typical default, though possibly a little high given small dataset
epochs <- 10

# Design your convolutional neural network - define how it is structured

# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = "softmax") 

# keras models are modified in place - hence %>% rather than <- 

print(model)

# Compile and train the model

# First compile the model
model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
) # defines what error term to use to measure the accuracy

# Now train the model with fit_generator
history <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
)
# Error messages on mac - trying on WVD




