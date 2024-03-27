import cv2
import face_recognition

# Load face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

def predict_nationality(ethnicity):
    # Mapping ethnicity to nationality
    nationality_mapping = {
        "Asian": "Japanese",  # Example mapping, you can add more mappings
        "Caucasian": "American",
        "Black": "Nigerian",
        "Hispanic": "Mexican",
        "Middle Eastern": "Saudi Arabian",
        "South Asian": "Indian",
        "Native American": "Navajo",
        "Pacific Islander": "Fijian"
        # Add more mappings here based on your knowledge or data
    }
    return nationality_mapping.get(ethnicity, "Unknown")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Ethnicity prediction
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_locations = [(y, x+w, y+h, x)]
        face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
        
        if len(face_encodings) > 0:
            # Only predict ethnicity if a face is detected
            ethnicity = face_recognition.face_distance([face_recognition.face_encodings(frame)[0]], face_encodings[0])
            if ethnicity[0] < 0.6:
                ethnicity_label = "Asian"  # Example ethnicity labels, you can add more labels
            else:
                ethnicity_label = "Caucasian"
            
            # Predict nationality based on ethnicity
            nationality = predict_nationality(ethnicity_label)
        else:
            nationality = "Unknown"
        
        label = f"{ethnicity_label}, {nationality}"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imshow("Ethnicity and Nationality Prediction", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
