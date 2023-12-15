import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import json as js

status=True

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic= mp.solutions.holistic # Mediapipe Solutions
#Drawing specs
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
#cap
cap = cv2.VideoCapture(0)
#initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  
    while status==True:
        ret, frame= cap.read()
        #Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Make detections
        results = holistic.process(image)
        image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Draw body pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),)
        #Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #Draw right hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        cv2.imshow('Holistic Model detection', image)
        print(results.pose_landmarks)
        if(cv2.waitKey(10) & 0xFF == ord('q')):
            break


cap.release()
cv2.destroyAllWindows()


