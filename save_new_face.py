# This file is for saving a new face in face_encodings dir.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) and dlib to be installed only to read from your webcam.

import cv2
import numpy as np
import dlib

def main(file, name, detect_faces, predictor):
    while True:
        # Grab a single frame from WebCam
    	ret, frame = cv2.imread(file)

    	# Find all the faces and face enqcodings in the frame
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	faces = detect_faces.detectMultiScale(gray, 1.3,5)
    	
    	face_locations = []
    	for (x,y,w,h) in faces:
    		temp_tuple = (y, x+w, y+h, x)
    		face_locations.append(temp_tuple)
    	
        # If there is atleast 1 face then do face recognition 
    	if (len(face_locations) > 0 and len(face_locations) < 2):
            for face_location in face_locations:
                face_encoding = predictor(frame, face_location)

                # Save encoding in a numpy file
                np.save("face_encodings/"+person_name+'.npy', face_encoding)
        else:
            print("Either no face or more then one face detected. Please check the image file again")
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="give file for to extract face from it", required=True)
    parser.add_argument("-n", "--name", type=str, help="give name for the person", required=True)

    args = parser.parse_args()

    file = args.file
    name = args.name

    detect_faces = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    main(file, name, detect_faces, predictor)