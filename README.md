# Identification-Apple-Diseases-through-Deep-CNN-architectures
In this work the design for classification of 3 different types of apple diseases and healthy apples, was made through Deep Learning methods. AlexNet, VGG, DenseNet, ResNet and Inception models were analyzed and compared. The best model DenseNet-121 provided 99.35% of accuracy.

Our best model DenseNet-121 converged the quickest and had the highest single-
run validation accuracy (98.05%) across trials spanning 1–15 epochs. Performance was
further improved to 99.35% overall accuracy, precision = 0.993, recall = 0.994, and F1
= 0.993 in a follow-up 30-epoch trial with early halting. Black rot and cedar rust have
excellent class-level recognition, with only five misclassifications overall, according to
the confusion matrix.

#Working environment, libraries and models

All of the process was run first, in personal laptop with GPU AMD Ryzen 5 4600H
and 16GB of RAM, and then in more powerful machine that SDU University pro-
vided, with GPU of Intel gen 12 i9-12900KF and 64GB of RAM. All of the code was
written in Python. TenserFlow and Keras libraries was used for building and training
neural network. They defined layers, optimizers, and training loops, making whole
prototype simple. ImageDataGenerator sublibrary of Keras, it simplifies the process
by producing batches of modelcompatible augmented image data directly. AlexNet,
ResNet50, DenseNet121 are pre-trained models. They save time and computational
resources by efficiently extracting characteristics from photos without requiring initial
training. We used each of them separately to compare and define which of them the
best fit for our case. Model layers were used for next part of preprocessing. Adam is
an optimizer from Keras library. It was used for adjusting learning rates adaptively for
each parameter. It is also suitable for sparse gradients and nonstationary objectives,
making it effective in transfer learning scenarios

#Data Acquisition and Preprocessing

Several CNN models were used for training DL model for identification apple diseases.
The dataset used for this purpose, consist a total number of 3171 apples leaf images.
This is subset of popular PlantVillage dataset, which consist 4 different classes, apples
infected with apple scab, black rot, cedar rust and healthy apples

#Model Training

First, by randomly changing a portion of input units to 0 during training, model
layers were utilized to create completely connected layers for classification, lowering
model complexity and preventing overfitting. Second, to capture intricate patterns,
1024 units of connected layers were then added using the Relu activation algorithm.
For multitask classifications, the last layer features four units and a softmax activation
function. The base model’s layers were frozen to avoid updating pre-trained layers
too soon. It is also helpful for transferring knowledge to a new assignment without
interfering with the features that ImageNet has learned. Next, we used the Adam
optimizer with a learning rate of 0.001 as DenseNet performed better with this setups
during experiments. This keeps the learning rates for each parameter distinct and
modifies them using estimations of the gradient’s mean (first moment) and variance
(second 42 moment). Faster and more steady convergence results from this. In this
instance, the low rate is appropriate. A generator that produces a single batch of
photos and their labels at once was then used. It specifies the number of batches the
model must process in order to finish an epoch. For example, if there are 1000 samples
and the batch size is 32, the model will process 31 full batches with 32 samples each
and for the last batch it will go with remaining 8 samples in each epoch (1000 = 8
(mod 32)). Next, the model was re-compiled for fine-tuning. This process is almost
identical to compiling process, except the learning rate was reduced to 0.00001 in
order to avoid making large changes to the pre-trained weights, which may destabilize
training. Finally, the model was finetuned for another 10 epochs. It is aimed that it will potentially yield better performance by refining the weights of the trainable layers.
