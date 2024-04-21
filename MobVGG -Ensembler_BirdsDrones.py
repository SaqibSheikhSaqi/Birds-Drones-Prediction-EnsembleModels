#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator





# In[ ]:


# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is NOT available.")


# In[ ]:


# Set random seed for reproducibility
tf.random.set_seed(42)

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 2  # Change this based on the number of classes in your dataset

# Load MobileNetV2 pre-trained on ImageNet
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load VGG16 pre-trained on ImageNet
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in both models
for layer in mobilenet_model.layers:
    layer.trainable = False

for layer in vgg16_model.layers:
    layer.trainable = False

# Combine MobileNetV2 and VGG16 models
input_tensor = Input(shape=(224, 224, 3))
mobilenet_output = mobilenet_model(input_tensor)
vgg16_output = vgg16_model(input_tensor)
combined_output = Concatenate()([mobilenet_output, vgg16_output])

# Add additional layers for classification
x = GlobalAveragePooling2D()(combined_output)
x = Dense(512, activation='relu')(x)
output_tensor = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the combined model
combined_model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)

# Compile the model
combined_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Display model summary
combined_model.summary()


# In[ ]:


# Use ImageDataGenerator for data augmentation and loading datasets
valid_path = '/content/drive/MyDrive/BirdsDrones/BirdsDronesDataset/test/'
train_path = '/content/drive/MyDrive/BirdsDrones/BirdsDronesDataset/train/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# For binary classification, set class_mode='binary'
test_generator = test_datagen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, batch_size=16, class_mode='binary')
train_generator = train_datagen.flow_from_directory(train_path, target_size=IMAGE_SIZE, batch_size=16, class_mode='binary')



# In[ ]:


# Train the model
history = combined_model.fit(train_generator, epochs=6, validation_data=test_generator)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pred = combined_model.predict(test_generator)
print(pred)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
#plt.savefig('LossVal_loss')


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train acccuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
#plt.savefig('LossVal_loss')


# In[ ]:


import os
import shutil
import numpy as np

origin = '/content/drive/MyDrive/BirdsDrones/BirdsDronesDataset/test/birds/'
target1 = '/content/drive/MyDrive/BirdsDrones/TestCheck/birds/'
files = os.listdir(origin)
actual=[]
predicted=[]
values=[]
imageName=[]
ac=[]
pr=[]
for file_name in files:
    print(origin+file_name)
    print(target1+file_name)
    shutil.copy(origin+file_name, target1+file_name)
    imageName.append(str(origin+file_name))
    print(str(origin+file_name))
    #print("Original" + str(origin+file_name))
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/BirdsDrones/TestCheck/',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'binary')
    images,cls = next(test_set)
    #print("Actual Child")
    pred = combined_model.predict(test_set)

    for my_array in pred:
      values.append(my_array)
      #print(np.argmax(my_array))
      max_index = np.argmax(my_array)
      print("Actual: 0 predicted: " + str(max_index))
      predicted.append(max_index)
      actual.append(0)
    path = r"/content/drive/MyDrive/BirdsDrones/TestCheck/birds/"
    for fileN in os.listdir(path):
        # construct full file path
        fi = path + fileN
        if os.path.isfile(fi):
            #print('Deleting file:', fi)
            os.remove(fi)


# In[ ]:



origin = '/content/drive/MyDrive/BirdsDrones/BirdsDronesDataset/test/drones/'
target1 = '/content/drive/MyDrive/BirdsDrones/TestCheck/drones/'
files = os.listdir(origin)
for file_name in files:
    #print(origin+file_name)
    #print(target1+file_name)
    shutil.copy(origin+file_name, target1+file_name)
    imageName.append(str(origin+file_name))
    print(str(origin+file_name))
    #print("Original" + str(origin+file_name))
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/BirdsDrones/TestCheck/',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'binary')
    images,cls = next(test_set)
    #print("Actual Middle")
    pred = combined_model.predict(test_set)

    for my_array in pred:
      values.append(my_array)
      print("Actual: 1 predicted: " + str(max_index))
      #print(np.argmax(my_array))
      max_index = np.argmax(my_array)
      predicted.append(max_index)
      actual.append(1)
    path = r"/content/drive/MyDrive/BirdsDrones/TestCheck/drones/"
    for fileN in os.listdir(path):
        # construct full file path
        fi = path + fileN
        if os.path.isfile(fi):
            print('Deleting file:', fi)
            os.remove(fi)


# In[ ]:


# Evaluate the model
test_loss, test_acc = combined_model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')


# In[ ]:


combined_model.save("/content/drive/MyDrive/BirdsDrones//Ensembler-BirdsDrones.h5")


# In[ ]:


np.save('/content/drive/MyDrive/BirdsDrones/HistoryEnsembler.npy', history.history)


# In[ ]:


import numpy as np
import pandas as pd
imageNameData=pd.DataFrame(imageName)
valuesData=pd.DataFrame(values)
predictedData=pd.DataFrame(predicted)
actualData=pd.DataFrame(actual)


# In[ ]:





# In[ ]:


imageNameData.to_csv('/content/drive/MyDrive/BirdsDrones/imageName-Ensembler.csv')
valuesData.to_csv('/content/drive/MyDrive/BirdsDrones/valuesData-Ensembler.csv')
predictedData.to_csv('/content/drive/MyDrive/BirdsDrones/predictedData-Ensembler.csv')
actualData.to_csv('/content/drive/MyDrive/BirdsDrones/actualData-Ensembler.csv')


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actual, predicted)


# In[ ]:


print(cm)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(actual, predicted))


# In[ ]:


### Confusion Matrix
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
#predictions = model.predict(x_test, steps=len(x_test), verbose=0)
#y_pred=model.predict(x_test)
#y_pred = np.round(y_pred)

y_pred = np.argmax(predicted, axis=-1)

y_true=np.argmax(actual, axis=-1)

cm = confusion_matrix(actual, predicted)

## Get Class Labels
#labels = le.classes_
class_names = [0,1]

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(5, 5))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(class_names, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Refined Confusion Matrix', fontsize=20)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(actual, predicted)


# In[ ]:


mae


# In[ ]:


mae_values = [abs(a - p) for a, p in zip(actual, predicted)]

# Create a histogram
plt.hist(mae_values, bins=10, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('Mean Absolute Error (MAE)')
plt.ylabel('Frequency')
plt.title('Histogram of MAE Values')

# Show the plot
plt.show()


# In[ ]:


import seaborn as sns
sns.histplot(mae_values, kde=True, bins=10, color='skyblue')

# Add labels and title
plt.xlabel('Mean Absolute Error (MAE)')
plt.ylabel('Density')
plt.title('Density Plot of MAE Values')

# Show the plot
plt.show()


# In[ ]:


len(predicted)


# In[ ]:


len(actual)


# In[ ]:





# In[ ]:


for i in range(0,2):
  print(i)


# In[ ]:





# In[ ]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Assuming 'actual' and 'predicted' are your actual labels and predicted probabilities
# Replace them with your actual data

# Define the ranges for each class
class_ranges = [(0, 425), (426, 1011), (648, 1033), (1034, 1371)]

# Binarize the labels
y_bin = label_binarize(actual, classes=[0, 1, 2, 3])

# Compute and plot ROC curve and AUC for each class
n_classes = y_bin.shape[1]

plt.figure(figsize=(12, 8))

for i in range(n_classes):
    start, end = class_ranges[i]

    # Check if there are positive instances for the current class
    if y_bin[start:end, i].sum() > 0:
        fpr, tpr, _ = roc_curve(y_bin[start:end, i], predicted[start:end])
        roc_auc = auc(fpr, tpr)

        plt.subplot(2, 2, i + 1)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Class {i}')
        plt.legend(loc='lower right')

plt.tight_layout()
plt.show()


# In[ ]:


# Assuming 'actual' and 'predicted' are your actual labels and predicted probabilities
# Replace them with your actual data

# Define the ranges for the first class
class_ranges = [(0, 413), (414, 647), (648, 1033), (1034, 1371)]

# Initialize arrays
a = []
p = []

# Append values for the first class
start, end = class_ranges[0]
a.extend(actual[start:end])
p.extend(predicted[start:end])

# Now 'a' and 'p' contain values for the first class


# In[ ]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming 'a' contains actual labels and 'p' contains predicted probabilities
# Replace them with your actual data

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(actual, predicted)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


# Check if there are positive instances for the current class
if y_bin[start:end, i].sum() > 0:
    fpr, tpr, _ = roc_curve(y_bin[start:end, i], predicted[start:end])
    roc_auc = auc(fpr, tpr)
print(roc_auc)
    # Rest of the plotting code


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # New Section
