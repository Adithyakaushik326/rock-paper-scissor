import numpy as np
import cv2
import tensorflow as tf
import random
import time
# from tensorflow import keras
import os
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from tensorflow.keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

my_score = 0 
comp_score=0

def select_winner(human,comp,my_score,comp_score):
    
    if human == comp:
        return 'draw'
    elif human=='rock':
        if comp =='scissors':
            return 'me'
        elif comp =='paper':
            return 'comp'
    elif human=='scissors':
        if comp =='paper':
            return 'me'
        elif comp =='rock':
            return 'comp'
    elif human=='paper':
        if comp =='rock':
            return 'me'
        elif comp =='scissors':
            return 'comp'
    
    

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=45,
#         zoom_range=0.2,
#         horizontal_flip=False,
#         width_shift_range=0.1,  
#         height_shift_range=0.1,
#         validation_split=0.2,
# )
# train_gen = train_datagen.flow_from_directory(
#     '/pic/',
#     target_size = (160,160),
#     batch_size=32,
#     class_mode = 'categorical',
#     shuffle=True,
# )

# val_gen = train_datagen.flow_from_directory(
#     '/pic/',
#     target_size=(224,224),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False,
#     subset = 'validation'
# )
model_base = MobileNet(weights='imagenet',include_top=False,input_shape=(160,160,3))
model = Sequential()
model.add(model_base)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
    
# history = model.fit_generator(
#     train_gen, 
#     steps_per_epoch  = 20, 
#     validation_data  = val_gen,
#     validation_steps = 20,
#     epochs = 5, 
#     verbose = 2
# )
# model_score = model.evaluate_generator(val_gen,steps=20)
# print("Model Test Loss:",model_score[0])
# print("Model Test Accuracy:",model_score[1])

model.load_weights('model.h5')
d = {0:'paper',1:'rock',2:'scissors'}
vid = cv2.VideoCapture(0) 
prev = "none"
t = time.localtime()
past = int(time.strftime("%S", t))
choice = ''
winner =''
img = cv2.imread('comp-pic/paper.png')
while(True): 
    t = time.localtime()
    present = int(time.strftime("%S", t))
    if present == 00:
        present=60
    ret, frame = vid.read() 
    cv2.rectangle(frame,(75,180),(235,340),(0,0,255),2)


    c = frame[180:340,75:235]        
    c = c/255.0
    c = np.reshape(c,(-1,160,160,3))
    
    pred  = model.predict(c)    
    # time.sleep(1)\
    move = d[np.argmax(pred)]
    if abs(present-past) >3:
        past = present
    if abs(present-past)==3:
        choice = random.choice(['comp-pic/paper.png','comp-pic/scissors.png','comp-pic/rock.png'])
        img = cv2.imread(choice)
        
        choice = choice.split('/')[1]
        choice = choice.split('.')[0]
        
        winner = select_winner(move,choice,my_score,comp_score)
        if winner == 'me':
            my_score+=1
        elif winner == 'comp':
            comp_score+=1
        past = present
        print(move)
    frame = cv2.putText(frame, f'Your move : {move}', (20,120), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.8, (0, 0, 255) , 2, cv2.LINE_AA) 

    frame = cv2.putText(frame, f'Comp move : {choice}', (400,120), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.7, (0, 0, 255) , 2, cv2.LINE_AA) 
    frame = cv2.putText(frame, f'winner : {winner}', (250,380), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.8, (255, 0, 0) , 2, cv2.LINE_AA) 
    frame[140:300,450:610] =img
    if abs(present-past)!=0:
        a = abs(present-past)
        if a==1:
            a=2
        else:
            a=1
        frame = cv2.putText(frame, f'{a}', (300,100), cv2.FONT_HERSHEY_SIMPLEX,  
                    3, (255, 0, 0) , 5, cv2.LINE_AA)
    frame = cv2.putText(frame, f'Score {my_score} : {comp_score}', (200,450), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.8, (0, 255, 0) , 2, cv2.LINE_AA)
    cv2.imshow('frame', frame) 
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  

vid.release() 
cv2.destroyAllWindows() 