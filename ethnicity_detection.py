import glob
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
COLS = ['Asian', 'White', 'Black']
N_UPSCLAE = 1
# ------- Ethnicity Prediction
from database import mydatabase
dbms = mydatabase.MyDatabase(mydatabase.SQLITE, dbname='/Users/divyachandana/Documents/NJIT/work/summertasks/may25-may30/Park_face/mydb.sqlite')

# images_path = '/Users/divyachandana/Documents/NJIT/work/summertasks/jun1-jun5/atlanta'

# images_path = '/Users/divyachandana/Documents/NJIT/work/summertasks/jun1-jun5/nyc'


def main():
	with open('face_model.pkl', 'rb') as f:
		clf, labels = pickle.load(f, encoding="latin1")
	# db_table = 'face_attributes_atlanta'
	db_table = 'face_attributes_nyc'

	# files = glob.glob(r'/Users/divyachandana/Documents/NJIT/work/summertasks/jun1-jun5/atlanta/*.jpg')
	files = glob.glob(r'/Users/divyachandana/Documents/NJIT/work/summertasks/jun1-jun5/nyc/**/*.jpg')
	newly_cropped = 0
	no_coordinates = 0
	skipped_images = 0
	for j,f in enumerate(files):

		filename = os.path.basename(f)
		# filename='beltline2415.jpg'
		# f = '/Users/divyachandana/Documents/NJIT/work/summertasks/jun1-jun5/atlanta/'+filename
		query = "select id,image_path,face_rectangle_top,face_rectangle_left,face_rectangle_width,face_rectangle_height \
		from {} where status is 'processed'  and ethnicity is 'NULL' and \
		image_path like '%{}%'".format(db_table,filename)
		result = dbms.get_face_attributes(query)
		# print(len(result))
		if len(result)>0:
			print(j)
			for i,r in enumerate(result):
				try:
					id = r[0]
					widthby2 = round(int(r[4]) / 2)
					heightby2 = round(int(r[5]) / 2)
					x1 = int(r[3]) - widthby2
					y1 = int(r[2]) - heightby2
					x2 = int(r[3]) + int(r[4]) + widthby2
					y2 = int(r[2]) + int(r[5]) + heightby2

					cords = (x1, y1, x2, y2)
					image_obj = Image.open(f)
					cropped_image = image_obj.crop(cords)
					# cropped_image_path = './cropped/{}_{}'.format(i,filename)
					cropped_image_path = './cropped_nyc/{}_{}'.format(i,filename)
					cropped_image.save(cropped_image_path)
				except:
					print("coordinates missing")
					no_coordinates+=1
				try:
					race = predict_one_image(cropped_image_path, clf, labels)
					# print(race)
					os.remove(cropped_image_path)
					u_query = "UPDATE {} set ethnicity={} WHERE id={}".format(db_table,"'"+race+"'",id)
					dbms.update_face_attribute(u_query)
					newly_cropped += 1
				except:
					print("Skipping {}".format(cropped_image_path))
					skipped_images+=1

				# if not race:
				# 	continue
				# os.remove()
		# else:
			# continue
	print('newly_cropped',newly_cropped)
	print('no_coordinates',no_coordinates)
	print('skipped_images',skipped_images)

	print("Finished")
	# 	break

def predict_one_image(img_path, clf, labels):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings = extract_features(img_path)
    if not face_encodings:
        return None, None
    pred = pd.DataFrame(clf.predict_proba(face_encodings),columns = labels)
    pred = pred.loc[:, COLS]
    maxValuesObj = pred.idxmax(axis=1)
    # print(maxValuesObj,'-----')
    # print(maxValuesObj[0])
	# res = max(pred[0:])
    return maxValuesObj[0]

def extract_features(img_path):
    """Exctract 128 dimensional features
    """
    X_img = face_recognition.load_image_file(img_path)
    locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE)
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    return face_encodings












if __name__ == "__main__":
    main()