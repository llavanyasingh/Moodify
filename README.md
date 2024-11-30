# Moodify
The code builds a mood-based music recommendation system using emotion detection via webcam. It analyzes the user's facial expression, predicts emotions, and recommends music tracks accordingly. The recommendations are displayed in a Streamlit app with clickable song links, artist names, and an engaging user interface.

## Features:
Real-time emotion detection using webcam input.
Emotion-based music recommendations from a dataset of tracks categorized by emotional tags.
Web interface built with Streamlit for an interactive user experience.
Music links to directly access tracks on external sites.

The interface looks like:
![image](https://github.com/user-attachments/assets/b7518b87-5606-43e8-8740-cc49bf90aea3)

To run on your system use these commands:

   For installation of necessary dependencies:
```bash
  pip install -r requirements.txt
```

   Move into desired directory (one with the code)
```bash
  cd C:\Users\user-name\Desktop\project
```
  To run the app
```bash
  streamlit run app.py
```

## Libraries used:
1- numpy: For numerical operations and handling arrays.

2- streamlit: For creating the web interface and displaying the app.

3- cv2 (OpenCV): For computer vision tasks like face detection and emotion recognition.

4- pandas: For data manipulation and handling the music dataset.

5- collections.Counter: For counting occurrences of emotions in the list.

6- tensorflow.keras: For building and using the CNN model for emotion detection.



