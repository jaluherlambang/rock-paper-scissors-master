import tensorflow as tf
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


model = tf.keras.models.load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)

prev_move = None

cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)  # Normal window mode
is_fullscreen = False  # Status layar

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (20, 150), (220, 350), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (400, 150), (600, 350), (255, 255, 255), 2)

    def preprocess_image(image):
        image = cv2.resize(image, (150,150))
        image = image /255.0
        return image

    # extract the region of image within the user rectangle
    roi = frame[20:220, 150:350]
    preprocessed_image = preprocess_image(roi)#img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (150, 150)) #227, 227 before

    input_image = np.reshape(preprocessed_image, (1, 150, 150, 3))

    # predict the move made
    pred = model.predict(input_image) #model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (0, 135), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer Move: " + computer_move_name,
                (390, 130), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (200, 430), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (200, 200)) #400
        #print(icon.shape)
        #print(frame[80:500, 240:1200].shape)
        print(frame.shape)
        frame[150:350, 400:600] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break
    elif k == ord('f') or k == ord('F'):
        # Toggle mode layar penuh saat tombol 'f' atau 'F' ditekan
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty("Rock Paper Scissors", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Rock Paper Scissors", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Rock Paper Scissors", 800, 600)  # Ganti ukuran sesuai kebutuhan
        cv2.imshow("Rock Paper Scissors", frame)   

cap.release()
cv2.destroyAllWindows()
