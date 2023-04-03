import sys
import time
import cv2
import numpy as np
import pandas as pd

sys.path.append('TF-Movenet/examples/lite/examples/pose_estimation/raspberry_pi')

from ml import Movenet
from ml import Classifier
from data import BodyPart
from data import Point
import utils
import tensorflow as tf
from tensorflow import keras

def get_center_point(landmarks, left_bodypart, right_bodypart):
    left = landmarks[:, left_bodypart.value]
    right = landmarks[:, right_bodypart.value]
    center = left * 0.5 + right * 0.5
    return center

def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.
    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = np.linalg.norm(shoulders_center - hips_center)

    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = np.expand_dims(pose_center_new, axis=1)

    # Broadcast the pose center to the same size as the landmark vector to perform substraction
    pose_center_new = np.broadcast_to(pose_center_new, (landmarks.shape[0], 17, 2))

    # Dist to pose center
    d = landmarks - pose_center_new
    dist_to_pose_center = d[0]

    # Max dist to pose center
    max_dist = np.max(np.linalg.norm(dist_to_pose_center, axis=1))

    # Normalize scale
    pose_size = max(torso_size * torso_size_multiplier, max_dist)
    return pose_size

def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
    """

    '''print("TYPE")
    print(type(landmarks))'''

    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = np.expand_dims(pose_center, axis=1)
    
    # Broadcast the pose center to the same size as the landmark vector to perform substraction
    pose_center = np.broadcast_to(pose_center, (landmarks.shape[0], 17, 2))
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    '''print("TYPE")
    print(type(landmarks))'''

    return landmarks

def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = np.reshape(landmarks_and_scores, (17, 3))

    # Normalize landmarks 2D
    #print(np.array([reshaped_inputs[:, :2]]))
    landmarks = normalize_pose_landmarks(np.array([reshaped_inputs[:, :2]]))
    #print(type(landmarks)) --> <class 'tensorflow.python.framework.ops.EagerTensor'>
    
    # Flatten the normalized landmark coordinates into a vector
    embedding = landmarks.flatten()
    return embedding

def normalize_person(person):
    
    landmarks = [[i.coordinate.x, i.coordinate.y, i.score] for i in person.keypoints]
    landmarks = [item for sublist in landmarks for item in sublist]

    landmarks_array = np.array(landmarks)
    embedding = landmarks_to_embedding(landmarks_array)
    return embedding.reshape(34)

def run(classification_model, label_file, estimation_model):

    # Load Movenet Model (Lightning or Thunder)
    pose_detector = Movenet(estimation_model)

    # Video Capture
    cap = cv2.VideoCapture("/Users/rahul/Desktop/Tested-Input/Handstand-Test-Videos/3.mp4")

    # Score Threshold Per Keypoint
    keypoint_detection_threshold_for_classifier = 0.1

    # Classification Model
    classifier = Classifier(classification_model, label_file)

    # Loop Frames and Run Inference
    while cap.isOpened():
        success, frame = cap.read()

        # Flip Frame (Recorded with Front Facing Camera)
        #frame = cv2.flip(frame, 1)
        # Isolate People from Frame
        list_persons = [pose_detector.detect(frame)]

        # Check All Keypoints Visible
        # Check Keypoint Score is Above Threshold
        person = list_persons[0]
        # Person - KeyPoint, X, Y, Score

        # Normalize pose, get center point, get pose size, normalize pose landmarks, landmarks to embedding, preprocess data
        person_numpy = normalize_person(person)

        min_score = min([keypoint.score for keypoint in person.keypoints])

        color = (255, 255, 255)

        # Error Message
        if min_score < keypoint_detection_threshold_for_classifier:
            print("Make sure the person is fully visible in the camera.")

        # Pose Classification
        else:
            prob_list = classifier.classify_pose(person_numpy)
            print(prob_list)

            top_class_name = prob_list[0].label
            print(top_class_name)

            if top_class_name == "Average":
                color = (0, 155, 255)
            elif top_class_name == "Good":
                color = (255, 255, 0)
            elif top_class_name == "Bad":
                color = (0, 0, 255)
            elif top_class_name == "Perfect":
                color = (0, 255, 0)

        frame = utils.visualize(frame, list_persons, color)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        cv2.imshow(estimation_model, frame)

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()


def main():
    #classifier = "./TF-Models/Handstand/Handstand-Classifier.tflite"
    classifier = "/Users/rahul/Documents/Calisthenics-App/TF-Models/Handstand/Handstand-Classifier.tflite"
    #classifier = "/Users/rahul/Documents/Calisthenics-App/Yoga-Test/pose_classifier/pose_classifier.tflite"
    #classifier = "/Users/rahul/Desktop/yoga_classifier.tflite"

    #label_file = "./TF-Models/Handstand/Handstand-Labels.txt"
    label_file = "/Users/rahul/Documents/Calisthenics-App/TF-Models/Handstand/Handstand-Labels.txt"
    #label_file = "/Users/rahul/Documents/Calisthenics-App/Yoga-Test/pose_classifier/pose_labels.txt"
    #label_file = "/Users/rahul/Documents/Calisthenics-App/Yoga-Test/pose_classifier/pose_labels.txt"

    estimation_model = "./TF-Movenet/movenet_lightning.tflite"

    run(classifier, label_file, estimation_model)


if __name__ == '__main__':
  main()