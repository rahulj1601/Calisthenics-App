import sys
import time
import cv2
import numpy as np

sys.path.append('TF-Movenet/movenet-models/examples/lite/examples/pose_estimation/raspberry_pi')

from ml import Classifier
from ml import Movenet
import utils

def run(classification_model, label_file, estimation_model):

    # Load Movenet Model (Lightning or Thunder)
    pose_detector = Movenet(estimation_model)

    # Video Capture
    cap = cv2.VideoCapture(0)

    # Score Threshold Per Keypoint
    keypoint_detection_threshold_for_classifier = 0.0

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

        keypoint_color = (255,255,255)
        edge_color = (255,255,255)

        # Error Message
        if min_score < keypoint_detection_threshold_for_classifier:
            print("Make sure the person is fully visible in the camera.")
        
        # Pose Classification
        else:
            prob_list = classifier.classify_pose(person)

            top_class_name = prob_list[0].label
            print(top_class_name)

            if top_class_name == "Average":
                keypoint_color = (0,155,255)
                edge_color = (0,155,255)
            elif top_class_name == "Good":
                keypoint_color = (0,255,0)
                edge_color = (0,255,0)
            elif top_class_name == "Bad":
                keypoint_color = (0,0,255)
                edge_color = (0,0,255)

        frame = utils.visualize(frame, list_persons, keypoint_color, edge_color)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        cv2.imshow(estimation_model, frame)

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()


def main():
    classifier = "pose_classifier"
    label_file = "pose_labels.txt"
    estimation_model = "movenet_lightning"

    run(classifier, label_file, estimation_model)


if __name__ == '__main__':
  main()