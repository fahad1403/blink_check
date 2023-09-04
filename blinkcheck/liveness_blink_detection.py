import cv2
import f_liveness_detection
import cv2
import imutils
import time


def challenge_result(question, blinks_up):
    if question == "blink eyes":
        if blinks_up == 1: 
            challenge = "pass"
        else:
            challenge = "fail"

    return challenge


def liveness_blink_check(video_path = "test_vid.mp4"):
    # COUNTER and TOTAL are used for blink counting
    COUNTER, TOTAL = 0, 0
    challenge_res1 = 'fail'
    blinks_up = 0
    frame_interval = 5

    # input_type decides whether to use a webcam or static images for input.
    input_type = "video"
    i = 1

    question = "blink eyes"

    print(f"\n\nQUESTION: {question}\n\n")

    #----------------------------- Video ------------------------------
    if input_type == "video":
        vid = cv2.VideoCapture(video_path)
        while True:
            start_time = time.time()
            ret, im = vid.read()
            if not ret:
                break

            if i % frame_interval == 0:
                im = imutils.resize(im, width=720)
                out = f_liveness_detection.detect_liveness(im, COUNTER, TOTAL)
                boxes = out['box_face_frontal']

                TOTAL = out['total_blinks']
                if TOTAL > 0:
                    blinks_up = 1

                COUNTER = out['count_blinks_consecutives']

                challenge_res1_temp = challenge_result(question, blinks_up)
                if challenge_res1=='fail' and challenge_res1_temp=='pass':
                    challenge_res1 = challenge_res1_temp

            end_time = time.time() - start_time
            FPS = 1 / end_time
            i+=1
        
        print(f"\n\nCHALLENGE RESULT : {challenge_res1}\n\n")
        if challenge_res1 == "pass":
            print("\n\n------------- SUCCESS --------------\n\n")
        else:
            print("\n\n------------- FAIL --------------\n\n")
            