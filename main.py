import os
from statistics import StatisticsError

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from random import choice, shuffle
from scipy import stats as st
from collections import deque


def gather_data(nsamples):
    global rock, paper, scissor, nothing
    cap = cv2.VideoCapture(0)
    trigger = False
    counter = 0
    box_size = 234
    width = int(cap.get(3))
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        if counter == nsamples:
            trigger = not trigger
            counter = 0
        cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 150), 2)
        cv2.namedWindow("Collecting images", cv2.WINDOW_NORMAL)
        if trigger:
            roi = frame[5: box_size - 5, width - box_size + 5: width - 5]
            eval(class_name).append([roi, class_name])
            counter += 1
            text = "Collected Samples of {}: {}".format(class_name, counter)
        else:
            text = "Press 'r' to collect rock samples, 'p' for paper, 's' for scissor and 'n' for nothing"
        cv2.putText(frame, text, (3, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Collecting images", frame)
        k = cv2.waitKey(1)
        if k == ord('r'):
            trigger = not trigger
            class_name = 'rock'
            rock = []
        if k == ord('p'):
            trigger = not trigger
            class_name = 'paper'
            paper = []
        if k == ord('s'):
            trigger = not trigger
            class_name = 'scissor'
            scissor = []
        if k == ord('n'):
            trigger = not trigger
            class_name = 'nothing'
            nothing = []
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


samples = 100
gather_data(samples)

labels = [tupl[1] for tupl in rock] + [tupl[1] for tupl in paper] + [tupl[1] for tupl in scissor] + [tupl[1] for tupl in
                                                                                                     nothing]
images = [tupl[0] for tupl in rock] + [tupl[0] for tupl in paper] + [tupl[0] for tupl in scissor] + [tupl[0] for tupl in
                                                                                                     nothing]
images = np.array(images, dtype="float") / 255.0
print('Total images: {} , Total Labels: {}'.format(len(labels), len(images)))

encoder = LabelEncoder()
Int_labels = encoder.fit_transform(labels)
one_hot_labels = to_categorical(Int_labels, 4)
(trainX, testX, trainY, testY) = train_test_split(images, one_hot_labels, test_size=0.25, random_state=50)
images = []

image_size = 224
N_mobile = tf.keras.applications.NASNetMobile(input_shape=(image_size, image_size, 3), include_top=False,
                                              weights='imagenet')
N_mobile.trainable = False
x = N_mobile.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(712, activation='relu')(x)
x = tf.keras.layers.Dropout(0.40)(x)
preds = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.models.Model(inputs=N_mobile.input, outputs=preds)
print("Layers in Model: {}".format(len(model.layers[:])))

augment = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    horizontal_flip=False,
    fill_mode="nearest"
)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 15
batchsize = 20
history = model.fit(x=augment.flow(trainX, trainY, batch_size=batchsize),
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // batchsize, epochs=epochs)


model.save("rps4.h5")
model = load_model("rps4.h5")

label_names = ['nothing', 'paper', 'rock', 'scissor']


def findout_winner(user_move, computer_move):
    if user_move == computer_move:
        return 'Tie'
    elif user_move == "rock" and computer_move == "scissor":
        return 'User'
    elif user_move == "rock" and computer_move == "paper":
        return 'Computer'
    elif user_move == "scissor" and computer_move == "rock":
        return 'Computer'
    elif user_move == 'scissor' and computer_move == "paper":
        return 'User'
    elif user_move == 'paper' and computer_move == 'rock':
        return 'User'
    elif user_move == 'paper' and computer_move == 'scissor':
        return 'Computer'


def show_winner(user_score, computer_score):
    if user_score > computer_score:
        img = cv2.imread("C:/Users/aarya/Desktop/userwins.png")
    elif user_score < computer_score:
        img = cv2.imread("C:/Users/aarya/Desktop/computerwins.png")
    else:
        img = cv2.imread("C:/Users/aarya/Desktop/tie.png")
    cv2.putText(img, "Press 'ENTER' to play again, else exit",
                (150, 530), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("Rock Paper Scissors", img)
    # If enter is pressed.
    k = cv2.waitKey(0)
    # If the user presses 'ENTER' key then return TRUE, otherwise FALSE
    if k == 13:
        return True
    else:
        return False


def display_computer_move(computer_move_name, frame):
    icon = cv2.imread("C:/Users/aarya/Desktop/pictures/{}.png".format(computer_move_name), 1)
    icon = cv2.resize(icon, (224, 224))
    roi = frame[0:224, 0:224]
    mask = icon[:, :, -1]
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    icon_bgr = icon[:, :, :3]
    img1_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(icon_bgr, icon_bgr, mask=mask)
    combined = cv2.add(img1_bg, img2_fg)
    frame[0:224, 0:224] = combined
    return frame


cap = cv2.VideoCapture(0)
box_size = 234
width = int(cap.get(3))
attempts = 5

computer_move_name = "nothing"
final_user_move = "nothing"
label_names = ['nothing', 'paper', 'rock', 'scissor']
computer_score, user_score = 0, 0
rect_color = (255, 0, 0)
hand_inside = False
total_attempts = attempts
confidence_threshold = 0.70
smooth_factor = 5
de = deque(['nothing'] * 5, maxlen=smooth_factor)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)
    roi = frame[5: box_size - 5, width - box_size + 5: width - 5]
    roi = np.array([roi]).astype('float64') / 255.0
    pred = model.predict(roi)
    move_code = np.argmax(pred[0])
    user_move = label_names[move_code]
    prob = np.max(pred[0])

    if prob >= confidence_threshold:
        de.appendleft(user_move)
        try:
            final_user_move = st.mode(de)[0][0]
        except StatisticsError:
            print('Stats error')
            continue
        if final_user_move != "nothing" and hand_inside == False:
            hand_inside = True
            computer_move_name = choice(['rock', 'paper', 'scissor'])
            winner = findout_winner(final_user_move, computer_move_name)
            display_computer_move(computer_move_name, frame)
            total_attempts -= 1

            if winner == "Computer":
                computer_score += 1
                rect_color = (0, 0, 255)
            elif winner == "User":
                user_score += 1;
                rect_color = (0, 250, 0)
            elif winner == "Tie":
                rect_color = (255, 250, 255)
            if total_attempts == 0:
                play_again = show_winner(user_score, computer_score)
                if play_again:
                    user_score, computer_score, total_attempts = 0, 0, attempts
                else:
                    break
        elif final_user_move != "nothing" and hand_inside == True:
            display_computer_move(computer_move_name, frame)
        elif final_user_move == 'nothing':
            hand_inside = False
            rect_color = (255, 0, 0)
# This is where all annotation is happening.
    cv2.putText(frame, "Your Move: " + final_user_move,
                (420, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (2, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Your Score: " + str(user_score),
                (420, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Computer Score: " + str(computer_score),
                (2, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Attempts left: {}".format(total_attempts), (190, 400), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                (100, 2, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (width - box_size, 0), (width, box_size), rect_color, 2)
    # Display the image
    cv2.imshow("Rock Paper Scissors", frame)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
