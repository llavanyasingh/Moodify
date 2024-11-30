# Importing necessary libraries and modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import base64

# Reading the music dataset
df = pd.read_csv("C:\\Users\\lavan\\Desktop\\coding\\Emotion-based-music-recommendation-system-main\\music_info.csv")

# Renaming columns to be more meaningful
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

# Selecting the relevant columns for analysis
df = df[['name','emotional','pleasant','link','artist']]
print(df)

# Sorting data based on emotional and pleasant values
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
print(df)

# Splitting the data based on different emotions
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Function that filters music based on the user's detected emotions
def fun(list):
    data = pd.DataFrame()

    # Logic for selecting tracks based on a single emotion
    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    # Logic for selecting tracks based on two emotions
    elif len(list) == 2:
        times = [30,20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
               data = pd.concat([df_sad.sample(n=t)])

    # Logic for selecting tracks based on three emotions
    elif len(list) == 3:
        times = [55,20,15]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    # Logic for selecting tracks based on four emotions
    elif len(list) == 4:
        times = [30,29,18,9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
               data = pd.concat([df_sad.sample(n=t)])

    # Logic for selecting tracks based on five emotions
    else:
        times = [10,7,6,5,2]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    print("data of list func... :",data)
    return data

# Function to process and return a list of unique emotions in order of occurrence
def pre(l):
    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    print("Processed Emotions:", result)

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
            print(result)
    print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul

# Building the CNN model for emotion detection
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load the pre-trained weights for emotion detection
model.load_weights('C:\\Users\\lavan\\Desktop\\coding\\Emotion-based-music-recommendation-system-main\\model.h5')

# Dictionary mapping the indices to emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Set up OpenCV to avoid using OpenCL
cv2.ocl.setUseOpenCL(False)

# Initialize webcam for emotion detection
cap = cv2.VideoCapture(0)

# Load the Haarcascade Classifier for face detection
print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier('C:\\Users\\lavan\\Desktop\\coding\\Emotion-based-music-recommendation-system-main\\haarcascade_frontalface_default.xml')
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

# Setting the background image for the Streamlit app
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    font-family: 'Arial', sans-serif;
    color: #FFFFFF;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Display the title and description for the app
st.markdown("<h2 style='text-align: center; color: #C71585; font-family: Times New Roman, serif; font-size: 50px;'><b>★ Moodify ★</b></h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FFB533; font-family: Times New Roman, serif; font-size: 40px;'><b>Music Tailored to Your Emotions</b></h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #FFB533 ; font-family: Times New Roman, serif; font-size: 25px;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)

# Create a three-column layout for the user interface
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])  # The middle column is given more space
list = []

# Place the "Scan Me" button in
with col3:
    if st.button('Scan Me'):
        count = 0
        list.clear()  # Clear the emotion list before starting

        # Your existing code for button action goes here
        while True:
            ret, frame = cap.read()  # Capture frame from webcam
            if not ret:
                break
            # Convert the image to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces using the Haarcascade classifier
            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            count = count + 1  # Increment the count of frames processed

            # Loop through each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]  # Extract region of interest (face)
                # Preprocess the face image for emotion prediction
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                # Predict the emotion using the pre-trained model
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))  # Get the emotion with the highest probability

                # Append the predicted emotion to the list
                list.append(emotion_dict[max_index])

                # Display the predicted emotion on the webcam feed
                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Show the webcam feed with the face rectangles and emotion labels
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

            # Exit if the user presses the 's' key
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
            # Stop after processing 20 frames
            if count >= 20:
                break
        cap.release()  # Release the webcam
        cv2.destroyAllWindows()  # Close the webcam feed window

        # Process the emotion list and remove duplicates while maintaining order
        list = pre(list)
        # Display a success message on Streamlit app
        st.success("Emotions successfully detected")

    # When emotions are detected, recommend tracks based on the detected emotions
new_df = fun(list)

# Empty space for layout
st.write("")

# Display a header for the recommended tracks
st.markdown(
    "<h5 style='text-align: center; color: #FFB533;font-family: Times New Roman, serif; '><b>For suggested tracks featuring artist names</b></h5>",
    unsafe_allow_html=True)
st.write(
    "---------------------------------------------------------------------------------------------------------------------")

# Try block to avoid errors when displaying recommendations
try:
    # Loop through the recommended data and display track information
    for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(30)):
        # Display the track name as a clickable link
        st.markdown("""
            <h4 style='text-align: center;'>
                <a href={style="color: #FF5733; font-family: Times New Roman, serif; font-size: 22px;"}>
                    {} - {}
                </a>
            </h4>
            """.format(l, i + 1, n), unsafe_allow_html=True)

        # Display the artist name below the track name
        st.markdown(
            "<h5 style='text-align: center; color: #FFB533; font-family: Times New Roman, serif; font-size: 18px;'><i>{}</i></h5>".format(
                a), unsafe_allow_html=True)

        # Display a separator between tracks
        st.write(
            "---------------------------------------------------------------------------------------------------------------------")
except:
    pass  # If there are any errors (e.g., no recommendations), silently pass
