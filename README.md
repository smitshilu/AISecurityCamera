# AISecurityCamera

This is a simple Security Camera example which detects motion and perform face recognition using OpenCV and dlib on Raspberry PI. 

Blog Post: Coming Soon

Video: Coming Soon

## Demo
Coming Soon

Required libraries on PI:
  1. dlib
  2. opencv
  3. numpy
  
 ## Getting Stared
 ```
 # clone this repo
git clone https://github.com/smitshilu/AISecurityCamera.git
cd AISecurityCamera

# Start the Camera feed using following commnad
python start.py

# If you want to save a new face encoding
python save_new_face.py -f image.jpg -n "Smit"
OR
python save_new_face.py -f image.png -n "Smit"
```
