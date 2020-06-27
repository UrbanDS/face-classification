# ------- Ethnicity Prediction
# from __future__ import print_function
import argparse
import os
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
# we are only going to use 4 attributes
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1
# ------- Ethnicity Prediction


def main():
	with open('face_model.pkl') as f:
		clf, labels = pickle.load(f)