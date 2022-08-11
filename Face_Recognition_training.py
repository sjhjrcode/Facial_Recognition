#Currently not finished

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import argparse
import imutils
import os
import glob
import time
import face_recognition
import cv2
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
input_string = []

while True:
    temp_input = (input("Enter User Name for Face Recognitoin. Type q when finised\n"))
    if temp_input=='q':
        break
    input_string.append(temp_input)


base_dir = "./dataset/"
if((os.path.exists(base_dir))):
    print("dataset file exists")
else:
    os.makedirs(base_dir)
#user_dataset = "dataset/Steven/User."
user_dataset = []
str_temp = ""
for string in input_string:
    #print(input_string[0])
    str_temp = string
    print(str_temp)
    temp_dir = base_dir+str_temp
    print(temp_dir)
    user_dataset.append(temp_dir)



for string2 in user_dataset:
    if((os.path.exists(string2))):
        print("dataset file exists")
    else:
        os.makedirs(string2)
#behavior is strange detecting faces but not taking pictures will try with a similar but diffrent method. 

vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("./opencv/data/haarcascades/haarcascade_frontalface_default.xml")
face_id = 1
count = 0
while(vid_cam.isOpened()):
    ret, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for string2 in user_dataset:
        for (x,y,w,h) in faces:
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            cv2.imwrite(string2+"/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('frame', image_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif count>100:
                break
    if count>100:
        break

vid_cam.release()
cv2.destroyAllWindows()
"""
    # Use `ImageDataGenerator` to rescale the images.
    # Create the train generator and specify where the train dataset directory, image size, batch size.
    # Create the validation generator with similar approach as the train generator with the flow_from_directory() method.
"""

# We are trying to minimize the resolution of the images without loosing the 'Features'
# For facial recognition, this seems to be working fine, you can increase or decrease it
IMAGE_SIZE = 224

# Depending upon the total number of images you have set the batch size
# I have 50 images per person (which still won't give very accurate result)
# Hence, I am setting my batch size to 5
# So in 10 epochs/iterations batch size of 5 will be processed and trained
BATCH_SIZE = 5

# We need a data generator which rescales the images
# Pre-processes the images like re-scaling and other required operations for the next steps
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2)

# We separate out data set into Training, Validation & Testing. Mostly you will see Training and Validation.
# We create generators for that, here we have train and validation generator.
# Create a train_generator
train_generator = data_generator.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training')

# Create a validation generator
val_generator = data_generator.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation')

# Triggering a training generator for all the batches
for image_batch, label_batch in train_generator:
    break

# This will print all classification labels in the console
print(train_generator.class_indices)

# Creating a file which will contain all names in the format of next lines
labels = '\n'.join(sorted(train_generator.class_indices.keys()))

# Writing it out to the file which will be named 'labels.txt'
with open('labels.txt', 'w') as f:
    f.write(labels)

"""
We have to create the base model from the pre-trained CNN
Create the base model from the **MobileNet V2** model developed at Google
and pre-trained on the ImageNet dataset, a large dataset of 1.4M images and 1000 classes of web images.
First, pick which intermediate layer of MobileNet V2 will be used for feature extraction. 
A common practice is to use the output of the very last layer before the flatten operation,
The so-called "bottleneck layer". The reasoning here is that the following fully-connected layers 
will be too specialized to the task the network was trained on, and thus the features learned by
These layers won't be very useful for a new task. The bottleneck features, however, retain much generality.
Let's instantiate an MobileNet V2 model pre-loaded with weights trained on ImageNet. 
By specifying the `include_top=False` argument, we load a network that doesn't include the
classification layers at the top, which is ideal for feature extraction.
"""

# Resolution of images (Width , Height, Array of size 3 to accommodate RGB Colors)
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2 Hidden and output layer is in the model The first
# layer does a redundant work of classification of features which is not required to be trained
# (This is also called as bottle neck layer)

# Hence, creating a model with EXCLUDING the top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Feature extraction
# You will freeze the convolution base created from the previous step and use that as a feature extractor
# Add a classifier on top of it and train the top-level classifier.

# We will be tweaking the model with our own classifications
# Hence we don't want our tweaks to affect the layers in the 'base_model'
# Hence we disable their training
base_model.trainable = False

# Add a classification head
# We are going to add more classification 'heads' to our model

# 1 -   We are giving our base model (Top Layer removed, hidden and output layer UN-TRAINABLE)

# 2 -   2D Convolution network (32 nodes, 3 Kernel size, Activation Function)
#       [Remember how Convolutions are formed, with role of kernels]
#       [ Kernels in Convents are Odd numbered]
#       [ Kernels are just a determinant which is multiplied with Image Matrix]
#       [They help in enhancement of features]

# 3 -   We don't need all nodes, 20% of nodes will be dropped out [probability of each "bad" node being dropped is 20%
#       Bad here means the nodes which are not contributing much value to the final output

# 4 -   Convolution and pooling - 2X2 matrix is taken and a pooling is done
#       Pooling is done for data size reduction by taking average

# 5 -   All above are transformation layers, this is the main Dense Layer
#       Dense layer takes input from all prev nodes and gives input to all next nodes.
#       It is very densely connected and hence called the Dense Layer.
#       We are using the Activation function of 'Softmax'
#       There is another popular Activation function called 'Relu'

model = tf.keras.Sequential([
    base_model,  # 1
    tf.keras.layers.Conv2D(32, 3, activation='relu'),  # 2
    tf.keras.layers.Dropout(0.2),  # 3
    tf.keras.layers.GlobalAveragePooling2D(),  # 4
    tf.keras.layers.Dense(3, activation='softmax')  # 5
])

# You must compile the model before training it.  Since there are two classes, use a binary cross-entropy loss.
# Since we have added more classification nodes, to our existing model, we need to compile the whole thing
# as a single model, hence we will compile the model now

# 1 - BP optimizer [Adam/Xavier algorithms help in Optimization]
# 2 - Weights are changed depending upon the 'LOSS' ['RMS, 'CROSS-ENTROPY' are some algorithms]
# 3 - On basis of which parameter our loss will be calculated? here we are going for accuracy
model.compile(optimizer=tf.keras.optimizers.Adam(),  # 1
              loss='categorical_crossentropy',  # 2
              metrics=['accuracy'])  # 3

# To see the model summary in a tabular structure
model.summary()

# Printing some statistics
print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

# Train the model
# We will do it in 10 Iterations
epochs = 10

# Fitting / Training the model
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator)

# Visualizing the Learning curves [OPTIONAL]
#
# Let's take a look at the learning curves of the training and validation accuracy/loss
# When using the MobileNet V2 base model as a fixed feature extractor.

'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
'''

"""## Fine tuning
In our feature extraction experiment, you were only training a few layers on top of an MobileNet V2 base model. The weights of the pre-trained network were **not** updated during training.
One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the pre-trained model alongside the training of the classifier you added. The training process will force the weights to be tuned from generic features maps to features associated specifically to our dataset.
### Un-freeze the top layers of the model
All you need to do is unfreeze the `base_model` and set the bottom layers be un-trainable. Then, recompile the model (necessary for these changes to take effect), and resume training.
"""

# Setting back to trainable
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model
# Compile the model using a much lower training rate.
# Notice the parameter in Adam() function, parameter passed to Adam is the learning rate
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-5),
              metrics=['accuracy'])

# Getting the summary of the final model
model.summary()
# Printing Training Variables
print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

# Continue Train the model
history_fine = model.fit(train_generator,
                         epochs=5,
                         validation_data=val_generator
                         )
# Saving the Trained Model to the keras h5 format.
# So in future, if we want to convert again, we don't have to go through the whole process again
saved_model_dir = 'save/fine_tuning.h5'
model.save(saved_model_dir)
print("Model Saved to save/fine_tuning.h5")

'''
#tf.saved_model.save(model, saved_model_dir)
converter = tf.lite.TFLiteConverter.from_keras_model_file(model)
#Use this if 238 fails tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
# Let's take a look at the learning curves of the training and validation accuracy/loss
# When fine tuning the last few layers of the MobileNet V2 base model and training the classifier on top of it.
# The validation loss is much higher than the training loss, so you may get some overfitting.
# You may also get some overfitting as the new training set is
# Relatively small and similar to the original MobileNet V2 datasets.

acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']
loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
'''

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
from tensorflow import keras
#detection_model = model_builder.build(model_config=configs['model'], is_training=False)

detection_model = keras.models.load_model("./save/fine_tuning.h5")

def convert_classes(classes, start=1):
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text

my_file = open("labels.txt", "r")

data = my_file.read()
data_into_list = data.split("\n")
print(data_into_list)
my_file.close()

txt = convert_classes(data_into_list)
print(txt)
with open('label_map.pbtxt', 'w') as f:
    f.write(txt)

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



category_index = label_map_util.create_category_index_from_labelmap('./label_map.pbtxt')


#cap.release()



# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
detections = detect_fn(input_tensor)


#from matplotlib import pyplot as plt


#tensorflow/core/kernels/data/generator_dataset_op.cc:108] Error occurred when finalizing GeneratorDataset 