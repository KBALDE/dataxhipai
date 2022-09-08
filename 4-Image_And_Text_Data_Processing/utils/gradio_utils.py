# feature extraction SIFT and ORB
import gradio as gr


#read 
import pandas as pd
import numpy as np

import requests
import pandas as pd
import nltk


import json


from PIL import Image
from io import BytesIO


from PIL import Image, ImageFilter, ImageOps
from matplotlib import pyplot as plt

from utils import yelp_app 

import cv2


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r
    
    
    
def callImageRestAPIurl(site):
    
    
    response = requests.get(site, auth=BearerAuth(yelp_app.apiKey))
    # get an image from a review id
    js=response.json()
    url=js['reviews'][0]['user']['image_url']
    
    return url



# api call

def apiRestQueryTest(site):
    response = requests.get(site, auth=BearerAuth(yelp_app.apiKey))
    return response.json()['reviews'][0]['text']


# api call and return an image from a given link site just for illustration

def callImageRestAPI(site):
    
    
    response = requests.get(site, auth=BearerAuth(yelp_app.apiKey))
    # get an image from a review id
    js=response.json()
    url=js['reviews'][0]['user']['image_url']
    
    #
    
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img.resize((300,300))









def draw_keyp_sift(imgPath):
    """
    Takes image path and return keypoints to be drawn
    
    """
    img = cv2.imread(imgPath)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create() # intantiate
    kp = sift.detect(gray,None) # keypoints
    img_sf=cv2.drawKeypoints(gray, kp, gray, color=(255, 0, 0), flags=0)
    return img_sf




def draw_keyp_sift_test(img):
    """
    Takes image path and return keypoints to be drawn
    
    """
    #img = cv2.imread(imgPath)
    img=np.array(img)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create() # intantiate
    kp = sift.detect(gray,None) # keypoints
    img_sf=cv2.drawKeypoints(gray, kp, gray, color=(255, 0, 0), flags=0)
    #axim=plt.imshow(img_sf)
    img_sf=Image.fromarray(np.array(img_sf))
    return img_sf


def draw_keyp_orb(imgPath):
    """
    Takes image path and return keypoints to be drawn
    
    """
    img = cv2.imread(imgPath)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.ORB_create() # intantiate
    kp = sift.detect(gray,None) # keypoints
    img_orb=cv2.drawKeypoints(gray, kp, gray, color=(255, 0, 0), flags=0)
    return img_orb


def draw_keyp_orb_test(img):
    """
    Takes image path and return keypoints to be drawn
    
    """
    #img = cv2.imread(imgPath)
    gray= cv2.cvtColor(np.array(img),cv2.COLOR_BGR2GRAY)
    sift = cv2.ORB_create() # intantiate
    kp = sift.detect(gray,None) # keypoints
    img_orb=cv2.drawKeypoints(gray, kp, gray, color=(255, 0, 0), flags=0)
    img_orb=Image.fromarray(img_orb)
    return img_orb