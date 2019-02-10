# AISecurityCamera

This is a simple Security Camera example which detects motion and perform face recognition using OpenCV and dlib on Raspberry PI. 

Blog Post: [Medium](https://towardsdatascience.com/make-your-own-smart-home-security-camera-a89d47284fc7)

Video: [YouTube](https://www.youtube.com/watch?v=bqIY0pOsZZ0)

## Demo
![Demo](https://github.com/smitshilu/AISecurityCamera/blob/master/Demo.gif)

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
