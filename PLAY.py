import numpy as np
import cv2
import random
import time


# font = cv2.FONT_HERSHEY_SIMPLEX 
# org = (50, 50) 
# fontScale = 1
# color = (255, 0, 0) 
# thickness = 2
# image = cv2.putText(frame, 'OpenCV', (130,75), cv2.FONT_HERSHEY_SIMPLEX,  
#                    1, (255, 0, 0) , 2, cv2.LINE_AA) 




vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read() 
    img = cv2.imread(random.choice(['comp-pic/paper.png','comp-pic/scissor.png','comp-pic/rock.png']))
    # img = cv2.imread('pic/paper/54.png')
    cv2.rectangle(frame,(75,140),(235,300),(0,0,255),2)
     
    c = frame[140:300,75:235]
    # cv2.rectangle(frame,(450,140),(610,300),(255,0,0),2)
    frame[140:300,450:610] =img
    frame = cv2.putText(frame, 'Your move : Scissors', (20,120), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (0, 0, 255) , 2, cv2.LINE_AA) 
    
    frame = cv2.putText(frame, 'Comp move : Paper', (400,120), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.7, (0, 0, 255) , 2, cv2.LINE_AA) 
    frame = cv2.putText(frame, 'Score 1-1', (250,400), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255, 0, 0) , 2, cv2.LINE_AA) 
    cv2.imshow('frame', frame)
    time.sleep(1)
    # c = c/255.0
    # c = np.reshape(c,(-1,160,160,3))
    
    # pred  = model.predict(c)
    # time.sleep(1)\
    # move = d[np.argmax(pred)]
    # if  move!= prev:
    #     print(move)
    # prev = move
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  

vid.release() 
cv2.destroyAllWindows() 

