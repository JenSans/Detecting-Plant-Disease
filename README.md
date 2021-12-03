# Detecting-Plant-Disease
Image classification project for detecting diseases in plants!

**Overview**

In recent decades, farmers have experienced devastating crop loss as a result of the changes in the environment; global warming, weather pattern changes, pest infestations, etc. According to PlantVillage [source](http://arxiv.org/abs/1511.08060), we need to increase food production globally by 70% to feed an expected population of 9 billion people. This project aims to predict whether a crop is healthy or not and aid in PlantVillage's mobile disease diagnostics system! Early detection of disease is key for preventing the loss of a crop. It is especially important to salvage crops in developing communities. 

**Business and Data Understanding**

This project analyses over 50,000 plant images of healthy and diseased crops. PlantVillage will use this model to continue to improve on their expert level crop disease diagnostics. 

These crops include: 

1. Apples 
2. Blueberries 
3. Cherries 
4. Corn
5. Grapes 
6. Oranges 
7. Peaches
8. Bell Peppers 
9. Potatoes
10. Raspberries 
11. Soybeans 
12. Squash 
13. Strawberries 
14. Tomatoes

In exploring the data, it was found that the images have an imbalance with about 73% of the images being diseased plants and 27% being healthy plants. The diseases included in this dataset are scabs, black rot, rust, powdery mildew, gray leaf spots, Northern Leaf Blight, Black Measles, Isariopsis Leaf Spot, citrus greening, bacterial spots, early blight, late blight, leaf scorch, leaf mold, target spots, yellow leaf curl virus, and tomato mosaic virus. 

Almost all of the images have a Width of 256 pixels and Height of 256 pixels. There are 3 different types of images represented in this data including color images, grayscale images, and segmented images. This gives the model many different options to train on. 

**Modeling**

Baseline Model: 

The baseline model is CNN that adds one Convolutional layer and augments the images with a horizontal flip. The accuracy comes out low at about 19%, leaving a lot of room for improvement in the modeling process. Due to limited computing power, I wanted to check on how a model would perform by keeping the epochs and steps per epoch on the lower side at 10 epochs and 20 steps per epoch. When visualizing the model's epochs and validation accuracy, a gradual improvement does show, so adding more epochs and steps per epoch may drastically improve this model's accuracy. 

The final model takes into account what was learned from the previous 3 models before it. More convolutional layers are added, the images were augmented with a vertical flip, and a slight increase in shear. 

**Model Results**
The baseline model's accuracy came out to 19%, which is not an ideal accuracy by any means, but it is a starting point. 
The second model increases to about 40% accuracy, a drastic improvement from the baseline. 
The third model increases again to 72% accuracy! Now we're getting somewhere. 
The fourth model 

Parameter Tuning: 

Adjustments were made to prevent overfitting using Dropout. Through the modeling process, more layers are added as well as epochs and steps per epoch to check on the slow improvement on the model's performnce with accuracy. Throughout the modeling process, I gradually add convolutional layers to the model to see it's affect on the model's performance. 

**Conclusion**

Future Work: 

In future work, I'd like to include more images in the dataset that are common in other parts of the world. The fruits and vegetables in this dataset are very common in North America, and I'd like to include images representing the vegetation that is common to the cultures in other countries other than North America. I'd also want to shoot for a more balanced dataset that contains more healthy plant images to help with that imbalance in healthy vs. diseased images. 

**Presentation Link:** 

**Repository Navigation**

```
├── README.md                                   <- The top-level README for reviewers of this project. 
├── notebook.ipynb                              <- Final Notebook
├── presentation.pdf                            <- PDF of the Canva presentation. 
└── utils.py                                    <- Helpful functions to use in final notebook
```
