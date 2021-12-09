# Detecting-Plant-Disease
Image classification project for detecting diseases in plants! To view all the visuals for this project, please visit my nbviewer notebook [here](https://nbviewer.org/github/JenSans/Detecting-Plant-Disease/blob/main/notebook.ipynb). 

<img src="https://github.com/JenSans/Detecting-Plant-Disease/blob/main/images/Screen%20Shot%202021-12-08%20at%209.24.11%20AM.png" width="250" height="250"> <img src="https://github.com/JenSans/Detecting-Plant-Disease/blob/main/images/Screen%20Shot%202021-12-08%20at%209.24.06%20AM.png" width="250" height="250"> <img src="https://github.com/JenSans/Detecting-Plant-Disease/blob/main/images/Screen%20Shot%202021-12-08%20at%209.24.00%20AM.png" width="250" height="250">

## Overview

In recent decades, farmers have experienced devastating crop loss as a result of the changes in the environment; global warming, weather pattern changes, pest infestations, etc. According to [PlantVillage](http://arxiv.org/abs/1511.08060), we need to increase food production globally by 70% to feed an expected population of 9 billion people. This project aims to predict whether a crop is healthy or not and aid in PlantVillage's mobile disease diagnostics system! Early detection of disease is key for preventing the loss of a crop. It is especially important to salvage crops in developing communities. 

## Business and Data Understanding

This project analyses over 50,000 plant images of healthy and diseased crops. PlantVillage will use this model to continue to improve on their expert level crop disease diagnostics. 

These crops include Apples, Blueberries, Cherries, Corn, Grapes, Oranges, Peaches, Bell Peppers, Potatoes, Raspberries, Soybeans, Squash, Strawberries, and Tomatoes. 

In exploring the data, it was found that the images have an imbalance with about 73% of the images being diseased plants and 27% being healthy plants. The diseases included in this dataset are scabs, black rot, rust, powdery mildew, gray leaf spots, Northern Leaf Blight, Black Measles, Isariopsis Leaf Spot, citrus greening, bacterial spots, early blight, late blight, leaf scorch, leaf mold, target spots, yellow leaf curl virus, and tomato mosaic virus. 

Almost all of the images have a Width of 256 pixels and Height of 256 pixels. There are 3 different types of images represented in this data including color images, grayscale images, and segmented images. This gives the model many different options to train on. 

## Modeling

Baseline Model: 

The baseline model is CNN that adds one Convolutional layer and augments the images with a horizontal flip. The accuracy comes out low at about 19%, leaving a lot of room for improvement in the modeling process. Due to limited computing power, I wanted to check on how a model would perform by keeping the epochs and steps per epoch on the lower side at 10 epochs and 20 steps per epoch. When visualizing the model's epochs and validation accuracy, a gradual improvement does show, so adding more epochs and steps per epoch may drastically improve this model's accuracy. 

The final model takes into account what was learned from the previous 3 models before it. More convolutional layers are added, the images were augmented with a vertical flip, and a slight increase in shear. 

## Results
The best performing model was the third model with an accuracy of 74%. When testing out the model's predictions on images it hadn't trained on, it was able to predict the disease or health of the plant accurate 74% of the time! 

Adjustments were made to prevent overfitting using Dropout. Through the modeling process, more layers were added as well as epochs and steps per epoch to check on the slow improvement on the model's performance with accuracy. Throughout the modeling process, I gradually add convolutional layers to the model to see its effect on the model's performance. 

## Conclusion

For next steps, I would recommend using this model as a tracking system for disease on a farm. Using technology like this may help prevent significant crop loss in the future and keep small farms vital in their communities. This can be especially important in communities that rely heavily on their small farms. 

In future work, I'd like to include more images in the dataset that are common in other parts of the world. The fruits and vegetables in this dataset are very common in North America, and I'd like to include images representing the vegetation that is common to the cultures in other countries other than North America. I'd also want to shoot for a more balanced dataset that contains more healthy plant images to help with that imbalance in healthy vs. diseased images. 

## Presentation Link:
[Presentation](https://www.canva.com/design/DAExxsiS2ko/jNUSPPhrECkXyaLIdymbYw/view?utm_content=DAExxsiS2ko&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink)

## Repository Navigation

```
├── README.md                                   <- The top-level README for reviewers of this project. 
├── notebook.ipynb                              <- Final Notebook.
├── presentation.pdf                            <- PDF of the Canva presentation. 
├── images                                      <- Images used for README.
├── environment                                 <- Environment requirements to run this notebook. 
└── utils.py                                    <- Helpful functions to use in final notebook.
```
