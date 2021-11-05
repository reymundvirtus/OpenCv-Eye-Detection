import cv2

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # for face
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') # for eyes

while True:
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # find faces in the frame
    for (x, y, w, h) in faces: # loop through the faces
        reg_of_interest_gray = gray[y:y + w, x:x + w]
        reg_of_interest_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(reg_of_interest_gray, 1.3, 5)
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            cv2.rectangle(reg_of_interest_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 2) # return of all eyes

    cv2.imshow('Frame', frame)

    # to quit press 'b'
    if cv2.waitKey(1) == ord('b'):
        break


capture.release()
cv2.destroyAllWindows()