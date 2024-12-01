# Importing necessary libraries and modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Reading the music dataset
df = pd.read_csv("C:\\Users\\lavan\\Desktop\\coding\\Emotion-based-music-recommendation-system-main\\music_info.csv")

# Renaming columns to be more meaningful
if 'lastfm_url' in df.columns:
    df['link'] = df['lastfm_url']
else:
    st.error("The 'lastfm_url' column is missing in the dataset. Please verify your dataset.")

if 'track' in df.columns:
    df['name'] = df['track']
else:
    st.error("The 'track' column is missing in the dataset. Please verify your dataset.")

df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

# Selecting the relevant columns for analysis
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(drop=True, inplace=True)

# Splitting the data based on different emotions
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]


# Function that filters music based on the user's detected emotions
def fun(list):
    data = pd.DataFrame()
    emotion_map = {
        'Neutral': df_neutral,
        'Angry': df_angry,
        'fear': df_fear,
        'happy': df_happy,
        'Sad': df_sad,
    }
    sample_sizes = {
        1: [30],
        2: [30, 20],
        3: [55, 20, 15],
        4: [30, 29, 18, 9],
        5: [10, 7, 6, 5, 2],
    }

    for emotion, size in zip(list, sample_sizes.get(len(list), [10])):
        data = pd.concat([data, emotion_map.get(emotion, df_sad).sample(n=size)], ignore_index=True)

    return data


# Function to process and return a list of unique emotions in order of occurrence
def pre(l):
    ul = []
    for x in l:
        if x not in ul:
            ul.append(x)
    return ul


# Building the CNN model for emotion detection
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Load pre-trained weights for emotion detection
model.load_weights('C:\\Users\\lavan\\Desktop\\coding\\Emotion-based-music-recommendation-system-main\\model.h5')

# Map indices to emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Set up OpenCV to avoid using OpenCL
cv2.ocl.setUseOpenCL(False)

# Load Haarcascade Classifier for face detection
face = cv2.CascadeClassifier(
    'C:\\Users\\lavan\\Desktop\\coding\\Emotion-based-music-recommendation-system-main\\haarcascade_frontalface_default.xml')
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

# Streamlit UI Setup
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    color: #FFFFFF;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #C71585;'>★ Moodify ★</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FFB533;'>Music Tailored to Your Emotions</h3>",
            unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
list = []

with col3:
    if st.button('Scan Me'):
        count = 0
        list.clear()
        cap = cv2.VideoCapture(0)  # Initialize the webcam

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            st.error("Webcam is not accessible. Please check your webcam settings.")
            st.stop()

        while count < 20:
            ret, frame = cap.read()  # Capture frame from the webcam
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                list.append(emotion_dict[max_index])

            count += 1

        cap.release()  # Release the webcam
        cv2.destroyAllWindows()  # Close any OpenCV windows

        if not list:
            st.warning("No emotions detected. Please try again.")
        else:
            list = pre(list)  # Process emotions
            st.success("Emotions successfully detected.")

new_df = fun(list)

if new_df.empty:
    st.error("Press Scan me for recommendations")
else:
    st.write("Preview of some recommended songs:")
    st.write(new_df[['name', 'artist']].head())  # Display only 'name' and 'artist' columns

    try:
        for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(30)):
            st.markdown(f"""
                <h4 style='text-align: center;'>
                    <a href="{l}" style="color: #FF5733;" target="_blank">{i + 1}. {n}</a>
                </h4>
            """, unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center; color: #FFB533;'><i>{a}</i></h5>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
