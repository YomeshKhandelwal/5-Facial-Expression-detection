# 5-Facial-Expression-detection

This Python script utilizes computer vision and deep learning techniques to predict several facial features in real-time using a webcam feed. The script predicts age, gender, ethnicity, nationality, and dominant emotion of faces detected in the video stream.

Features Predicted:

1) Age and Gender Prediction:
The script detects faces in the video feed and predicts the age and gender of each detected face.
Age is classified into predefined age groups, and gender is categorized as male or female.

2) Ethnicity and Nationality Prediction:
Using facial recognition techniques, the script predicts the ethnicity and nationality of detected faces.
Currently, the script distinguishes between Asian and Caucasian ethnicities and predicts corresponding nationalities based on this classification.

3) Emotion Detection:
The script analyzes the dominant emotion expressed by each detected face using the DeepFace library.
Emotions detected include happiness, sadness, anger, surprise, fear, disgust, and neutrality.
