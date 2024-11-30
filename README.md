# Moodify
The code builds a mood-based music recommendation system using emotion detection via webcam. It analyzes the user's facial expression, predicts emotions, and recommends music tracks accordingly. The recommendations are displayed in a Streamlit app with clickable song links, artist names, and an engaging user interface.

The interface looks like:
![image](https://github.com/user-attachments/assets/b7518b87-5606-43e8-8740-cc49bf90aea3)

To run on your system use this command:
```bash
  streamlit run app.py
```

Libraries used:
numpy: For numerical operations and handling arrays.
streamlit: For creating the web interface and displaying the app.
cv2 (OpenCV): For computer vision tasks like face detection and emotion recognition.
pandas: For data manipulation and handling the music dataset.
collections.Counter: For counting occurrences of emotions in the list.
tensorflow.keras: For building and using the CNN model for emotion detection.



