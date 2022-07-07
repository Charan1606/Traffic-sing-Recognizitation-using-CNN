import cv2
import numpy as np
from keras.models import load_model

#lables(signs) list
signs = [ "Speed Limit 20 km/h", "Speed Limit 30 km/h", "Speed Limit 50 km/h",
            "Speed Limit 60 km/h", "Speed Limit 70 km/h", "Speed Limit 80 km/h",
            "End of Speed Limit 80 km/h", "Speed Limit 100 km/h", "Speed Limit 120 km/h",
            "No passing", "No passing for vechiles over 3.5 metric tons",
            "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
            "No vechiles", "Vechiles over 3.5 metric tons prohibited", "No entry", "General caution",
            "Dangerous curve to the left", "Dangerous curve to the right", "Double curve", "Bumpy road",
            "Slippery road", "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians",
            "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
            "End of all speed and passing limits", "Turn right ahead", "Turn left ahead", "Ahead only",
            "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory", 
            "End of no passing", "End of no passing by vechiles over 3.5 metric tons" ]

#parameters 
threshold = 0.75
brightness = 180
font1 = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN
frameHeight = 480
frameWidth = 640

#initializing camera
cam = cv2.VideoCapture(0)
cam.set(3, frameWidth)
cam.set(4, frameHeight)
cam.set(10, brightness)

#trained model
model = load_model("trainedModel1.h5")

#preprocessing image
def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

 
while True:
 
    # READ IMAGE
    success, imgOrignal = cam.read()
    
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    cv2.imshow("Processed Cam Feed", img)
    img = img.reshape(1, 32, 32, 1)
    
    # PREDICT IMAGE
    predictions = model.predict(img)
    print(predictions)
    signIndx = np.argmax(predictions)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        #printing data on screen
        signStr = "SIGN : [" + str(signIndx) + "] " + signs[signIndx]
        probabilityStr = "PROBABILITY : " + str(round(probabilityValue * 100, 2)) + "%"

        cv2.putText(imgOrignal, signStr, (20, 35), font1, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, probabilityStr, (20, 75), font1, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Live Camera Feed", imgOrignal)
    
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break
    