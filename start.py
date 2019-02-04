# This is a demo of running face recognition on a Raspberry Pi.
# This program will print out the names of anyone it recognizes to the console.

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.

import time
import cv2
import numpy as np
import glob
import os
import dlib
import RPi.GPIO as GPIO
import time

def motion_detected():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.IN)         #Read output from PIR motion sensor
    while True:
        i=GPIO.input(11)
        if i==0:                 #When output from motion sensor is LOW
            print ("No intruders")
            return False
        else:
            return True

def main(video_capture, saved_face_encodings, names, detect_faces, predictor, facerec):
    while True:
        # Grab a single frame from WebCam
    	ret, frame = video_capture.read()

    	# Find all the faces and face enqcodings in the frame
    	#face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	faces = detected_faces.detectMultiScale(gray, 1.3,5)
    	
    	face_locations = []
    	for (x,y,w,h) in faces:
    		temp_tuple = (y, x+w, y+h, x)
    		face_locations.append(temp_tuple)
    	
        # If there is atleast 1 face then do face recognition 
    	if (len(face_locations) > 0):
            for face_location in face_locations:
                face_encoding = predictor(frame, face_location)

                # See if the face is a match for the known face(s)
                match = list((np.linalg.norm(saved_face_encodings - face_encoding, axis=1) <= 0.6))
                
                name = "Unknown"
                for i, face in enumerate(match):
                    if (face):
                	    name = names[i]

                # Save image for future use
                if name=="Unknown":
                    cv2.imwrite(os.path.join(os.getcwd(), "unknown", str(int(time.time()))+".png"), frame)

		        # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

			    # Draw a label with a name below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left, bottom + 26), font, 1.0, (255, 255, 255), 1)

if __name__ == '__main__':
    detect_faces = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 320)
    video_capture.set(4, 240)

    saved_face_encodings = []
    names = []

    for face in glob.glob(os.path.join(os.getcwd(), "face_encodings", "*.npy")):
    	temp = np.load(face)[0]
    	saved_face_encodings.append(temp)
    	names.append(os.path.basename(face[:-4]))
    
    while True:
        if (motion_detected()):
            main(video_capture, saved_face_encodings, names, detect_faces, predictor, facerec)

    video_capture.release()