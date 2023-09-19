#Real time Finger Count
#use the hand tracking and the hand landmarks to count fingers
# 1 / install opencv and mediapipe
# 2 / import libraries

import cv2   #the opencv libraby
import time
import os    #to store the images

#import our module of handtracking

import HandTrackingModule as htm
wCam, hCam = 640, 480

# 3 / prepare camera settings

cap = cv2.VideoCapture(0)   #turn on our webcam

#giving size of the camera

cap.set(3, wCam)   #width
cap.set(4, hCam)   #height

# 4 / we have to take the images from the folder and store them so that we can display them later

folderPath = "FingerImages"

#list all the files present in FingerImages

myList = os.listdir(folderPath)
print(myList)

#create list of images
#overlay this image on our main image

overlayList = []

#loop through our list

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')

#save our image in the list

    overlayList.append(image)

#display the number of images

print(len(overlayList))

#initialize the previous time

pTime = 0

#create an object from our class with a 0.75 detection confidence

detector = htm.handDetector(detectionCon=0.75)

#4 for the thumb , 8 for the index , 12 for the middle finger , 16 for the ring finger

tipIds = [4, 8, 12, 16, 20]

# 5 / display finger counts with images

while True:
    # this will read our frame

    success, img = cap.read()

    # we want our detector to detect and find the hands
    #this returns draws on our hand

    img = detector.findHands(img)

    #we want it to find the position within our image
    #draw = false : because we are already drawing we don't want to draw again

    lmList = detector.findPosition(img, draw=False)

    #print(lmList)
    #we will try to get the tip of our fingers and based on that tip we can decide if our fingers our open or close

    if len(lmList) != 0:
        fingers = []

        #thumb
        #id number 1 : the x axis

        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #for the 4 fingers for right hand

        for id in range(1, 5):
            #exemple if number id=8 is under id=6 then it is open

            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                #if the finger is open then 1
                fingers.append(1)
            else:
                #if finger is closed
                fingers.append(0)

                #print(fingers)
                #print("Index finger open")

        #count the number of values of the number 1 present in the list fingers[]
        totalFingers = fingers.count(1)
        print(totalFingers)

        #if it is 1 it will take 0 and will take the first image in the folder , if it is 2 it will become 1 and it will take the second image
        #when the value of total fingers = 0 then it will give the value of -1 and in python list[-1] is the last element so it will take the 6th image

        h, w, c = overlayList[totalFingers-1].shape

        #define the size of images we want to overlay on the original image

        img[0:h, 0:w] = overlayList[totalFingers-1]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0, cv2.FILLED))

        #we will write the fps on our image with [(45,375)] is the region where we want to write

        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    #define the current time
    cTime = time.time()
    fps = 1/(cTime-pTime)
    #update the previous time
    pTime = cTime

    cv2.putText(img, f'FPS : {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 153), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1) #define the delay (= 1 ms) so that we can see our images
