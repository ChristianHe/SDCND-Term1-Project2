# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python len() and unique() api to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Refer to the "/Distribution of train set.png" for the exploratory visualization of the data set. From this graph, we can see that the number of images of different classes varies, which may effect the recognition result of certain traffic signs in real world. Augmentation of these images may help a lot.

Also, images from the train, validation and test set are showed with there labels.

In addition, showing image for each classes may help building sensitive cognition of German Traffic Signs. For those people who are not farmiliar with German Traffic Sign, like me, it may help a lot when testing on 5 new images.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the gray image has enough details regarding to the recognition with LeNet. And grayscale requires fewer parameters to be trained. 

As a second step, I normalized the image data for the convinence of SGD.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 5x5	    | 5x5 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 	    			|
| Convolution 1x1	    | 1x1 stride, valid padding, outputs 5x5x30     |
| RELU					|												|
| Fully connected		| Dropout 0.5, outputs 240       				|
| RELU					|												|
| Fully connected		| Dropout 0.5, outputs 84       				|
| RELU					|												|
| Output				| outputs 43        							|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer which realized the Momentum method, it will help training more fast and stable. Batch size is set to 128, epochs to 30, and learning rate initialized with 0.0005, decay every epoches. Here is the formula of learning rate decay:
rate = 0.0005 - (0.000005 * epoch) 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.964
* test set accuracy of 0.932

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Since the LeNet has a good performance recognizing the handwriting 0~9, which has 10 classes. A deeper or wider version of LeNet may also perform not too badly when recognizing the traffic sign with 43 classes.  

To deeper the LeNet but do not introduce too much parameters, 1*1 convolution layer is employed before the Fully connected layer. As the traffic signs have more characters than the 0~9, the number of convolution filters in the first convolution layer is increased to 10. Also the dropout is adapted both in Fully connnected layer 1 and layer2.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Refer to the "/testset2" for the five German traffic signs that I found on the web. All these 5 images have clearly background, already sized in 32x32x3 which can be feed directly into the model. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| Animal crossing		| Animal crossing								|
| No entrance			| No entrance									|
| Yield 	      		| Yield     					 				|
| Road work 			| Road work          							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.2%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For All the five images, the model is pretty sure that they are the right traffic sign with the probability of approximately 1.0. As for the the Keep right, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .970         			| Keep right   									| 
| .020     				| Dangerous curve to the right					|
| .005					| Go straight or right							|
| .004	      			| End of no passing				 				|
| .001				    | Slippery Road      							|

In addition, another 5 images were also downloaded and predicted, which you can find in the "/testset". These five images have different sizes, and the image Road Work also has a tree in its background.
The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares not so good to the accuracy on the test set of 93.2%.
The 2 images which were wrongly predicted are No vehicles and Turn Right. No Vehicles was predicted as Vehicles over 3.5 metric tons prohibited, and Turn Right was predicted as Stop. 
No Vehicles and Vehicles over 3.5 metric tons prohibited were similar, reducing the size of convolution filter from 5x5 to 3x3 or using multiscale convolution network may get more details and help distinguish the No Vehicles from Vehicles over 3.5 metric tons prohibited. Turn Right is not in the classes, so it was acceptable to deem it as Stop. Road Work with tree was correctly predicted with probobility of 0.5.

###Writing in the End
Overfitting and Underfitting are talking about the training set and validation set. It has nothing to do with the new images. Bad prediction on the new images might be the problem of the image or the preprocessing of it. Techeniques or methods used in this project are mainly concerned on these two things, visualization the accuracy of training and validation using curves may help figure out which techeniques or methods are useful. A beautiful curve shall be the accuracy of both training and validation are as much higher as possible, and the distance between training accuracy curve and validation curve also as closer as possible.