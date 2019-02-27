import cv2
import numpy as np
import pyautogui
import imutils
import time
from trex_classes import *
import mss
from qlearn import *
from matplotlib import pyplot as plt

click_replay = (683, 374)
#game_space = (371,275,610,160)
game_enviroment = {"top": 275, "left":371, "width": 610, "height": 160}
sct = mss.mss()

X_DISTANCES = 20
Y_DISTANCE = 70
X_OBSTACLE = 0
Y_OBSTACLE = 1
Y_AGENT = 2

def screenShot():
	image = pyautogui.screenshot(region=game_enviroment)
	image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#gray = cv2.GaussianBlur(gray, (21, 21), 0)
	return image

def fastScreenShot():
	image = np.asarray(sct.grab(game_enviroment))
	#image = cv2.cvtColor(np.asarray(sct.grab(game_space)), cv2.COLOR_RGB2BGR)
	return image

def gray(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def isAgent(w,h):
    return ((w==Agent.standing[2] and h==Agent.standing[3]) or (w==Agent.crouching[2] and h==Agent.crouching[3]))

class Enviroment:

    def restart_game():
        pyautogui.click(click_replay)

    def step(action):
        if(action == 0):
            Actions.jump()
        elif (action == 1):
            Actions.crouching()
        else:
            time.sleep(0.02)

    def get_state():
        frame = fastScreenShot()
        image = gray(frame)

        bin_image, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, 0, (0,255,0), 3)

        left_edges = []
        y_nearest = 10000
        x_nearest = 10000
        #70
        trex_y = 0
        win_objects = 0
        for c in contours:

            (x, y, w, h) = cv2.boundingRect(c)

            if (w > 60 or w < 10 or h < 18):
                continue
            if  (x==294 and y==88 and w==36 and h==32):
                States.game_over = True

            if x < Agent.position[0] and isAgent(w,h) and h<65:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
                trex_y = y
                continue
            if x < Agent.position[0] and not isAgent(w,h):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
                win_objects+=1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            left_edges.append(x)
            if x_nearest > x:
                x_nearest = x
                y_nearest = y

        if(trex_y < 107):
            trex_y = 0
        else: #When jump
            trex_y = 1

        if len(left_edges) >= 1: #States.obstacles.append([x, y, w, h])
            x_nearest = np.min(left_edges)
            if y_nearest <= Y_DISTANCE:
                y_pos = 2
            else:
                y_pos = 0
            States.nearest_obstacle = (int(x_nearest/X_DISTANCES),y_pos,trex_y)
        else:
            States.nearest_obstacle = (550,0,trex_y)

        cv2.imshow('Captured', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(1)

        return States.nearest_obstacle, win_objects



Enviroment.restart_game()

cv2.namedWindow('Captured', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('Captured', 359, 482)

action = 3

qlearn = Qlearn(122)#62

current_state, win_objects = Enviroment.get_state()
cont = 0
s_time = time.time()
fps = 0
while True:
    fps += 1
    if time.time() - s_time > 1:
        print("FPS: ", fps)
        fps = 0
        s_time = time.time()
    
    if States.game_over:
        Enviroment.restart_game()
        States.game_over = False
        cont+=1

    action = qlearn.getAction(current_state)
    Enviroment.step(action)

    next_state, next_win_objects = Enviroment.get_state()

    if (States.game_over):
        print('GAME OVER')
        reward = -10
    elif(win_objects < next_win_objects):
        reward = 10
        if (current_state[Y_AGENT] == 1 and action!=1):
            reward += 2
        else:
            reward -= 2
    else:
        if (current_state[X_OBSTACLE]>7 and action !=2): # if my object is far and my agent is walking
            reward = -1
        elif (current_state[X_OBSTACLE]<5 and action !=2):
            reward = 1
        elif ((current_state[Y_AGENT] == 1 or current_state[Y_OBSTACLE] == 0) and action==1): #if my agent is crouching when is high or my obstacle is low
            reward = -5
        elif(current_state[Y_OBSTACLE]==1 and action==1):
            reward = 5
        elif (current_state[Y_AGENT] == 1 and action==2 and current_state[X_OBSTACLE]<4):
            reward = 2
        else:
            reward = 1

    old_qvalue = qlearn.get_qvalue(current_state, action)
    next_max = qlearn.getMax(next_state)
    win_objects = 0

    qlearn.set_new_qvalue(old_qvalue, reward, next_max, current_state, action)

    current_state = next_state

cv2.waitKey(0)
cv2.destroyAllWindows()
cont