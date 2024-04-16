
import cv2
import numpy as np
import supervision as sv
import torch


class Detector:
    def __init__ (self, resolution: tuple[int,int], zone: np.ndarray ):
        self.resolution = resolution
        self.zone = zone
        self.model =  torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.model.classes = [0]
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=4)
        self.zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.GREEN, \
                                                      thickness=2, text_thickness=1, text_scale=1 )
    
    def detect(self, frame)->bool:
        '''
        Detects people in the zone

        args:
            frame:  list of pixels in format BGR (B-blue, G-green, R-red) 
                    in size of resolution[0] multiplicated by resolution[1]
        return:
                was a person detected in the zone

        '''
        frame = cv2.resize(frame, self.resolution)
        results = self.model(frame)
        detections = sv.Detections.from_yolov5(results)
        detections = detections[detections.confidence > 0.4]
        trig_arr = self.zone.trigger(detections=detections)
        people = np.sum(trig_arr == True)

        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.zone_annotator.annotate(scene=frame)

        return (people > 0)
    





