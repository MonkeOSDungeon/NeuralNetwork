import queue
import cv2
import os
import threading
import numpy as np
import supervision as sv
import torch
from time import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage



def send_email(to_email:str, image_path:str,from_email:str, server:smtplib.SMTP_SSL)-> None:
    '''
    sends email with text and image

    args: 
        to email: reciever email
        image_path: path to image
        from email: sender email
        server: smtp server
    return:
        none
    '''
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "" 
    # Add in the message body
    message_body = "" 
    # add image
    message.attach(MIMEText(message_body, 'plain')) 
    with open(image_path, 'rb') as file:
        image = MIMEImage(file.read())
        image.add_header('Content-Disposition', 'attachment', filename='image.jpg')
        message.attach(image)
    server.sendmail(from_email, to_email, message.as_string())

def get_zone(*args: tuple)-> np.ndarray:
    '''
    Makes zone from corner coordinates
    args: (x,y) coordinates
    return: array of zone corners
    '''
    polygon = []
    for arg in args:
        polygon.append(arg)
    return np.array(polygon)

def send_email_thread(frame_queue: queue, reciever: str, mail_info: tuple)->None:
    '''
    Saves image and sends it in thread

    args: 
        frame_queue: queue of frames where people were detected
        reciever: reciever email
        mail_info: sender email and server
    return: 
        none
    '''
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        cv2.imwrite('frame.jpg', frame) 
        send_email(reciever, 'frame.jpg', *mail_info) 
        os.remove('frame.jpg')  


def detect(video_path:str, zone:np.ndarray, reciever:str, mail_info:tuple)->None:
    '''
    Detects people in frames and sends email using the second frame

    args: 
        vide_path: path to video to open
        zone: array of polygon corners
        reciever: reciever email
        mail_info: sender email and server
    return: none
    
    '''
    # initiate polygon zone
    video_info = sv.VideoInfo.from_video_path(video_path)
    zone = sv.PolygonZone(polygon=zone, frame_resolution_wh=video_info.resolution_wh)
    # initiate model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.classes = [0]
    # initiate annotators
    box_annotator = sv.BoundingBoxAnnotator(thickness=4)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.GREEN, thickness=2, text_thickness=1, text_scale=1 )


    # initialize video capture
    cap = cv2.VideoCapture(video_path) 
    height = video_info.resolution_wh[1]
    width = video_info.resolution_wh[0]
    startTime = 0
    prev_det_time = 0 #previous detection in zone
    # people in the zone
    people = 0
    prev_det = 0 

    # creating a queue to transfer frames to the email sending thread
    frame_queue = queue.Queue()

    # creating and starting an email sending thread
    email_thread = threading.Thread(target=send_email_thread, args=(frame_queue, reciever, mail_info))
    email_thread.start()
    while cap.isOpened():
        currentTime = time()
        fps  = 1/(currentTime - startTime)
        startTime = currentTime
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (1280,720)) # change resolurion to hd
        # frame = cv2.resize(frame, (int(width*0.5),int(height*0.5)))# изменение размеров вручную для сохранения пропорций
        
        # detect
        results = model(frame)
        detections = sv.Detections.from_yolov5(results)
        detections = detections[detections.confidence > 0.4]
        trig_arr = zone.trigger(detections=detections)
        prev_det = people
        people = np.sum(trig_arr == True)
        if people>prev_det and prev_det==0: # if person detected in the zone
            detection_time = time()
            if detection_time-prev_det_time >3:
                frame_copy = frame.copy() 
                frame_queue.put(frame_copy)
                prev_det_time = detection_time 


        # annotate
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        # display the annotated frame
        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
        cv2.imshow("Real-time Video", frame)

        if cv2.waitKey(1) & 0xFF == 27: #нажать esc чтобы закрыть
            break
    
    frame_queue.put(None)
    email_thread.join()
    # release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def log_in():
    '''
    connects to email server

    return: touple(sender email, smtp server)
    '''
    password = "" # password to email
    sender = ""  # email from which emails wil be sent
    server = smtplib.SMTP_SSL("smtp.gmail.com",465)
    server.login(sender, password)# login to email
    return sender, server

def main():
    zone = get_zone((960,0),(500,0),(500,720),(960,720))
    video_path = "" 
    reciever = "" # reciever email
    detect(video_path, zone, reciever, log_in())

if __name__ == "__main__":
    main()
