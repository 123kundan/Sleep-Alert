import face_recognition
import cv2,os
from scipy.spatial import distance as dist
import winsound
import numpy as np

min_aer=0.30
eye_ar_cosec_frames=10

counter=0
alarm_on=False


def eye_aspect_ratio(eye):
    a=dist.euclidean(eye[1],eye[5])
    b=dist.euclidean(eye[2],eye[5])
    c=dist.euclidean(eye[0],eye[3])
    ear=(a+b)/(2*c)
    return ear

def main():
    global counter,alarm_on
    video_capture=cv2.VideoCapture(0)
    video_capture.set(3,320)
    video_capture.set(4,240)
    while True:
        ret,frame=video_capture.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        for face_landmark in face_landmarks_list:
            leftEye = face_landmark['left_eye']
            righteye = face_landmark["right_eye"]

            leftear=eye_aspect_ratio(leftEye)
            rightear=eye_aspect_ratio(righteye)
            ear=(leftear+rightear)/2

            lpts=np.array(leftEye)
            rpts=np.array(righteye)

            cv2.polylines(frame,[lpts],True,(255,255,0),1)
            cv2.polylines(frame,[rpts],True,(255,255,0),1)

            if ear<min_aer:
                counter+=1
                if counter>=eye_ar_cosec_frames:
                    if not alarm_on:
                        alarm_on=True
                        winsound.Beep(1000,2000)
                cv2.putText(frame,"alert! wakeup DANGER !!!!",(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

            else:
                counter=0
                alarm_on=False
            cv2.putText(frame,"Ear{:.2f}".format(ear),(300,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
            cv2.imshow("sleep detection",frame)
        if cv2.waitKey(1)==ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


