#read a json file with the coordinates of the landmarks

import json
import numpy as np
import matplotlib.pyplot as plt

#open the json file
with open('points.json') as f:
    data = json.load(f)

#extract the coordinates of the landmarks
landmark = data['Landmarks']
landmark = np.array(landmark)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0, len(landmark)):
    x= landmark[i]['x']
    y= landmark[i]['y']
    z= landmark[i]['z']
    # np.reshape(x,y,z,(x2,y2)
    #get the coordinates of the landmarks and the number of the landmark
    ax.scatter(x,y,z, c='r', marker='o')


    ax.text(x,y,z, i)
    # ax.text(x2,y2, i)
ax.scatter(0,0,0, c='b', marker='o')
# draw x,y,z axis
ax.plot([0,1],[0,0],[0,0], c='r')
ax.plot([0,0],[0,1],[0,0], c='g')
ax.plot([0,0],[0,0],[0,1], c='b')
plt.show()