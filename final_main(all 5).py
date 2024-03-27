import cv2
import face_recognition

from deepface import DeepFace

# Load face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_age_gender(frame, faceNet, ageList, genderList, padding=20):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100,300,0), 1)
            
            # Predict age and gender
            face = frame[max(0,y1-padding):min(y2+padding,frame.shape[0]-1),max(0,x1-padding):min(x2+padding, frame.shape[1]-1)]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (227, 227))
            blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            label = "{}, {}".format(gender, age)
            cv2.rectangle(frame, (x1, y1-30), (x2, y1), (0,255,0), -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
    return frame, bboxs

def predict_ethnicity_nationality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_locations = [(y, x+w, y+h, x)]
        face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
        
        if len(face_encodings) > 0:
            ethnicity = face_recognition.face_distance([face_recognition.face_encodings(frame)[0]], face_encodings[0])
            if ethnicity[0] < 0.6:
                ethnicity_label = "Asian"
            else:
                ethnicity_label = "Caucasian"
            
            nationality = predict_nationality(ethnicity_label)
        else:
            nationality = "Unknown"
        
        label = f"{ethnicity_label}, {nationality}"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
    return frame

def predict_nationality(ethnicity):
    nationality_mapping = {
        "Asian": "Japanese",
        "Caucasian": "American",
        "Black": "Nigerian",
        "Hispanic": "Mexican",
        "Middle Eastern": "Saudi Arabian",
        "South Asian": "Indian",
        "Native American": "Navajo",
        "Pacific Islander": "Fijian"
    }
    return nationality_mapping.get(ethnicity, "Unknown")

# Load age and gender models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

# Define age and gender lists
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    
    # Predict age and gender
    frame, bboxs = predict_age_gender(frame, faceNet, ageList, genderList)
    
    # Predict ethnicity and nationality
    frame = predict_ethnicity_nationality(frame)
    
    # Predict emotion
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    for result in results:
        emotion = result['dominant_emotion']
        cv2.putText(frame, emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2, cv2.LINE_4)
    
    cv2.imshow("Prediction", frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()