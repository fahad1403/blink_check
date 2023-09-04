import cv2
import f_utils
import dlib
import numpy as np
from f_blink_detection import eye_blink_detector


# instaciar detectores
frontal_face_detector    = dlib.get_frontal_face_detector()
blink_detector           = eye_blink_detector() 



def detect_liveness(im,COUNTER=0,TOTAL=0):
    
    gray = gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # face detection
    rectangles = frontal_face_detector(gray, 0)
    boxes_face = f_utils.convert_rectangles2array(rectangles,im)
    if len(boxes_face)!=0:
        # use only the biggest face
        areas = f_utils.get_areas(boxes_face)
        index = np.argmax(areas)
        rectangles = rectangles[index]
        boxes_face = [list(boxes_face[index])]

        # -------------------------------------- blink_detection ---------------------------------------
        '''
        input:
            - imagen gray
            - rectangles
        output:
            - status: "ok"
            - COUNTER: COUNTER and TOTAL are used for blink counting
            - TOTAL
        '''
        COUNTER,TOTAL = blink_detector.eye_blink(gray,rectangles,COUNTER,TOTAL)
    else:
        boxes_face = []
        TOTAL = 0
        COUNTER = 0

    # -------------------------------------- output ---------------------------------------
    output = {
        'box_face_frontal': boxes_face,
        'total_blinks': TOTAL,
        'count_blinks_consecutives': COUNTER
    }
    return output

