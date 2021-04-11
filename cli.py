'''
    Project Title:      COVID-19 Classification on X-ray images using Multimodal CNN's
    
    Group Members:      Milan Ashvinbhai Bhuva  -   IIT2018176
                        Manav Kamlesh Agrawal   -   IIT2018178
                        Mohammed Aadil          -   IIT2018179

	Run Instructions : 	$] python3 cli.py

'''

print('Importing the headers ...')
# basic ML
import os
import cv2
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # simply supress the warnings


# Sklearn
from sklearn.decomposition import PCA
from sklearn import svm

# tf and keras
from tensorflow.keras.applications import VGG16, EfficientNetB3, ResNet50
from keras.models import Sequential
from tensorflow.keras.models import Model
from keras.models import load_model

labels = [0, 1]

def pass_through_VGG(img):
    vgg_features = vgg16.predict(img)
    PCA_features = vgg_pca.transform(vgg_features)
    probs = vgg_svm.predict_proba(PCA_features)
    
    return probs


def pass_through_ResNet(img):
    resnet_features = resnet.predict(img)
    PCA_features = resnet_pca.transform(resnet_features)
    probs = resnet_svm.predict_proba(PCA_features)
    
    return probs


def pass_through_EffNet(img):
    effnet_features = effnet.predict(img)
    PCA_features = effnet_pca.transform(effnet_features)
    probs = effnet_svm.predict_proba(PCA_features)
    
    return probs

print('Loading the models ...')
# loading VGG16, ResNet50 and EffecientNetB3 pretrained models
vgg16 = load_model('H5/vgg_model.h5', compile=False)
resnet = load_model('H5/resnet_model.h5', compile=False)
effnet = load_model('H5/effnet_model.h5', compile=False)

# loading PCA models for VGG16, ResNet50 and EffecientNetB3
vgg_pca = pickle.load(open('models/vgg_pca.pkl', 'rb'))
resnet_pca = pickle.load(open('models/resnet_pca.pkl', 'rb'))
effnet_pca = pickle.load(open('models/effnet_pca.pkl', 'rb'))

# loading SVM models for VGG16, ResNet50 and EffecientNetB3
vgg_svm = pickle.load(open('models/vgg_svm.pkl', 'rb'))
resnet_svm = pickle.load(open('models/resnet_svm.pkl', 'rb'))
effnet_svm = pickle.load(open('models/effnet_svm.pkl', 'rb'))


print('\n\nEnter the address of image : ', end='')
path = input()
# importing the test image
img = cv2.imread(path)

# resize and reshape for VGG16, ResNet50 and EffecientNetB3
vgg16_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
vgg16_img = vgg16_img.reshape(1,224, 224,3)
resnet50_img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
resnet50_img = resnet50_img.reshape(1,512,512,3)
effnetb3_img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
effnetb3_img = effnetb3_img.reshape(1,300,300,3)

# passing through the 3 layres of each pretrained model
vgg_probs = pass_through_VGG(vgg16_img)
resnet_probs = pass_through_ResNet(resnet50_img)
effnet_probs = pass_through_EffNet(effnetb3_img)

# fusion using voting
merged = (effnet_probs + resnet_probs + vgg_probs)/3
merged_preds = np.argmax(merged, axis=1)

# printing the class
print("COVID" if labels[merged_preds[0]] else "Non-COVID")