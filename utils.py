from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D,Activation,LeakyReLU,BatchNormalization,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import cv2
import os
from os import listdir
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, LSTM, AveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from timeit import default_timer as timer
from pathlib import Path
import imagesize
from PIL import Image


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def get_roots(root_, root2_, root3_, folder_list): 
    # Get the color root 
    all_color = []
    for x in folder_list:     
            all_color.append(f'{root_}{x}/')
        
    # Get the grayscale root
    all_gray = []
    for y in folder_list:     
            all_gray.append(f'{root2_}{y}/')
            
    # Get the segmented root
    all_seg = []
    for z in folder_list:     
            all_seg.append(f'{root3_}{z}/')
            
    
    return [all_color, all_gray, all_seg]


def healthy_images(color, gray, seg): 
    
    healthy_col = []
    for x in color: 
        if x[-8:] == 'healthy/': 
            healthy_col.append(x)

    # Check the grayscale image labels    
    healthy_gray = []
    for y in gray: 
        if y[-8:] == 'healthy/': 
            healthy_gray.append(y)

    # Check the segmented image labels  
    healthy_seg = []
    for z in seg: 
        if z[-8:] == 'healthy/': 
            healthy_seg.append(z)
            
    return [healthy_col, healthy_gray, healthy_seg]


def all_images(col_, gray_, seg_): 
    
    #List of color images
    total_col_ = []
    for q in range(38):
        for img1 in Path(col_[q]).iterdir(): 
            if img1.suffix == ".JPG": 
                total_col_.append(f'{img1}')
        
    # List of grayscale images
    total_gray_ = []
    for r in range(38):
        for img2 in Path(gray_[r]).iterdir(): 
            if img2.suffix == ".JPG": 
                total_gray_.append(f'{img2}')
                
    # List of segmented images 
    total_seg_ = []
    for s in range(38):
        for img3 in Path(seg_[s]).iterdir(): 
             if img3.suffix == ".jpg": 
                total_seg_.append(f'{img3}')
    
    return [total_col_, total_gray_, total_seg_]


def image_sizes(root_col, root_gray, root_seg): 
    
    #Get sizes of color images
    image_sizes_col_ = []
    for a in range(len(root_col)):
        im = Image.open(root_col[a])
        image_sizes_col_.append(im.size)
    
    #Get sizes of grayscale images
    image_sizes_gray_ = []
    for b in range(len(root_gray)):
        im2 = Image.open(root_gray[b])
        image_sizes_gray_.append(im2.size)
        
    #Get sizes of segmented images
    image_sizes_seg_ = []
    for c in range(len(root_seg)):
        im3 = Image.open(root_seg[c])
        image_sizes_seg_.append(im3.size)
    
    return [image_sizes_col_, image_sizes_gray_, image_sizes_seg_]


def create_df(col_sizes, col_roots, gray_sizes, gray_roots, seg_sizes, seg_roots): 
    # Create a DataFrame with image Width, Height, FileName, and Type. 
    # Color first 
    col_image_df = pd.DataFrame(col_sizes, columns=('Width', 'Height'))
    col_image_df['FileName'] = col_roots
    col_image_df['Type'] = col_image_df['FileName'].str.contains('healthy')
    col_image_df['Type'] = col_image_df['Type'].astype('object')
    col_image_df['Type'].replace(True, 'Healthy', inplace=True)
    col_image_df['Type'].replace(False, 'Disease', inplace=True)
    
    # Grayscale DataFrame
    gray_image_df = pd.DataFrame(gray_sizes, columns=('Width', 'Height'))
    gray_image_df['FileName'] = gray_roots
    gray_image_df['Type'] = gray_image_df['FileName'].str.contains('healthy')
    gray_image_df['Type'] = gray_image_df['Type'].astype('object')
    gray_image_df['Type'].replace(True, 'Healthy', inplace=True)
    gray_image_df['Type'].replace(False, 'Disease', inplace=True)


    # Segmented DataFrame
    seg_image_df = pd.DataFrame(seg_sizes, columns=('Width', 'Height'))
    seg_image_df['FileName'] = seg_roots
    seg_image_df['Type'] = seg_image_df['FileName'].str.contains('healthy')
    seg_image_df['Type'] = seg_image_df['Type'].astype('object')
    seg_image_df['Type'].replace(True, 'Healthy', inplace=True)
    seg_image_df['Type'].replace(False, 'Disease', inplace=True)

    #Concatenate DataFrames
    return pd.concat([col_image_df, gray_image_df, seg_image_df], axis=0)


def plot_results(estimator_, model_, X, Y): 
    acc = estimator_.history['accuracy']
    val_acc = estimator_.history['val_accuracy']
    loss = estimator_.history['loss']
    val_loss = estimator_.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()

    plt.figure()

    # Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

    """Evaluating model accuracy by using the `evaluate` method"""

    print("[INFO] Calculating model accuracy")
    scores = model_.evaluate(X, Y)
    print(f"Test Accuracy: {round(scores[1]*100)}%")

def predict_disease(image_path, estimator):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float16) / 255.0
    np_image = np.expand_dims(np_image,0)
    plt.imshow(plt.imread(image_path))
    result = estimator.predict_classes(np_image)
    print((label_binarizer.classes_[result][0]))