import sys
import time
import cv2
import numpy as np

sys.path.append('TF-Movenet/examples/lite/examples/pose_estimation/raspberry_pi')

from ml import Movenet
from ml import Classifier
import utils

def run(classification_model, label_file, estimation_model):

    # Load Movenet Model (Lightning or Thunder)
    pose_detector = Movenet(estimation_model)

    # Video Capture
    cap = cv2.VideoCapture("/Users/rahul/Documents/CS310-App/input/Tested-Input/Handstand-Test-Videos/3.mp4")

    # Score Threshold Per Keypoint
    keypoint_detection_threshold_for_classifier = 0.1

    # Classification Model
    classifier = Classifier(classification_model, label_file)

    # Loop Frames and Run Inference
    while cap.isOpened():
        success, frame = cap.read()

        # Flip Frame (Recorded with Front Facing Camera)
        frame = cv2.flip(frame, 1)
        # Isolate People from Frame
        list_persons = [pose_detector.detect(frame)]

        # Check All Keypoints Visible
        # Check Keypoint Score is Above Threshold
        person = list_persons[0]
        # Person - KeyPoint, X, Y, Score

        min_score = min([keypoint.score for keypoint in person.keypoints])

        color = (255, 255, 255)

        # Error Message
        if min_score < keypoint_detection_threshold_for_classifier:
            print("Make sure the person is fully visible in the camera.")

        # Pose Classification
        else:
            prob_list = classifier.classify_pose(person)
            print(prob_list)

            top_class_name = prob_list[0].label
            print(top_class_name)

            if top_class_name == "Average":
                color = (0, 155, 255)
            elif top_class_name == "Good":
                color = (0, 255, 0)
            elif top_class_name == "Bad":
                color = (0, 0, 255)

        frame = utils.visualize(frame, list_persons, color)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        cv2.imshow(estimation_model, frame)

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()


def main():
    classifier = "./TF-Models/Handstand/Handstand-Classifier.tflite"
    label_file = "./TF-Models/Handstand/Handstand-Labels.txt"
    estimation_model = "./TF-Movenet/movenet_lightning.tflite"

    run(classifier, label_file, estimation_model)


if __name__ == '__main__':
  main()