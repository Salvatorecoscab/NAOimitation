import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import json as js
import numpy as np
status=True

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic= mp.solutions.holistic # Mediapipe Solutions
#Drawing specs
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
#cap
cap = cv2.VideoCapture(0)
Landmarksread={}
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
        # #when pressing a key the program will print the pose_landmarks
        # if(cv2.waitKey(10) & 0xFF == ord('g')):
        #     Landmarksread=results.pose_landmarks
        if(cv2.waitKey(10) & 0xFF == ord('g')):
            Landmarksread=results.pose_landmarks
            print(Landmarksread)
            
               
        if(cv2.waitKey(10) & 0xFF == ord('q')):
            Landmarksread=results.pose_landmarks
            break
#extract the coordinates of the landmarks


landmark = Landmarksread.landmark[:]
landmark = np.array(landmark)
print(landmark)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(0, len(landmark)):
    x= landmark[i].x
    y= landmark[i].y
    z = landmark[i].z

    #get the coordinates of the landmarks and the number of the landmark
    ax.scatter(x,y,z, c='r', marker='o')



    # ax.text(x,y,z, i)
ax.scatter(0,0,0, c='b', marker='o')
# draw x,y,z axis
ax.plot([0,1],[0,0],[0,0], c='r')
ax.plot([0,0],[0,1],[0,0], c='g')
ax.plot([0,0],[0,0],[0,1], c='b')
plt.show() 


cap.release()
cv2.destroyAllWindows()


