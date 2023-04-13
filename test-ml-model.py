###########################################################
# Test out produced ML Models before integrating into App #
###########################################################

# Module Imports
import sys
import time
import cv2
import numpy as np
 
# Movenet Imports
sys.path.append('TF-Movenet/examples/lite/examples/pose_estimation/raspberry_pi')
from ml import Movenet
from ml import Classifier
from data import BodyPart
from data import Point
import utils

# Get the center point between two given bodyparts
def get_center_point(landmarks, left_bodypart, right_bodypart):
    left = landmarks[:, left_bodypart.value]
    right = landmarks[:, right_bodypart.value]
    center = left * 0.5 + right * 0.5
    return center

# Get the size of the pose by marking distances between different body parts (e.g. hips, shoulders, torso)
def get_pose_size(landmarks, torso_size_multiplier=2.5):
    # Find center of hips
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

    # Find center of shoulders
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
    
    # Find the size of the torso
    torso_size = np.linalg.norm(shoulders_center - hips_center)

    # Find the central point of the body
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = np.expand_dims(pose_center_new, axis=1)
    pose_center_new = np.broadcast_to(pose_center_new, (landmarks.shape[0], 17, 2))
    
    # Calculate the pose size
    d = landmarks - pose_center_new
    dist_to_pose_center = d[0]
    max_dist = np.max(np.linalg.norm(dist_to_pose_center, axis=1))
    pose_size = max(torso_size * torso_size_multiplier, max_dist)

    return pose_size

# Normalise the pose
def normalize_pose_landmarks(landmarks):

    # Obtain the pose center
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = np.expand_dims(pose_center, axis=1)
    pose_center = np.broadcast_to(pose_center, (landmarks.shape[0], 17, 2))
    landmarks = landmarks - pose_center

    # Scale the pose to normalise it
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks

# Converting array of landmarks and scores to normalised flattened numpy array
def landmarks_to_embedding(landmarks_and_scores):
    reshaped_inputs = np.reshape(landmarks_and_scores, (17, 3))
    landmarks = normalize_pose_landmarks(np.array([reshaped_inputs[:, :2]]))
    embedding = landmarks.flatten()
    return embedding

# Normalise the person shrinking image but maintaining relative size
def normalize_person(person):

    # Extract x, y coordinates, and scores for each landmark
    landmarks = [[i.coordinate.x, i.coordinate.y, i.score] for i in person.keypoints]
    # Flatten the nested list into a one-dimensional list
    landmarks = [item for sublist in landmarks for item in sublist]

    # Convert the flattened landmark list into a NumPy array
    landmarks_array = np.array(landmarks)
    
    # Call a function to generate an embedding from the landmarks array
    embedding = landmarks_to_embedding(landmarks_array)
    
    # Reshape the embedding into a one-dimensional array of size 34
    return embedding.reshape(34)

def run(classification_model, label_file, estimation_model, video):

    # Load Movenet Model (Lightning or Thunder)
    pose_detector = Movenet(estimation_model)

    # Video Capture
    cap = cv2.VideoCapture(video)

    # Score Threshold Per Keypoint
    keypoint_detection_threshold_for_classifier = 0.1

    # Classification Model
    classifier = Classifier(classification_model, label_file)

    # Loop Frames and Run Inference
    while cap.isOpened():
        success, frame = cap.read()

        # Exit the loop once the video is finished
        if frame is None: break

        # Flip Frame (uncomment if required)
        # frame = cv2.flip(frame, 1)

        # Isolate People from Frame
        list_persons = [pose_detector.detect(frame)]

        person = list_persons[0]
        # Person - KeyPoint, X, Y, Score

        # Normalize pose, get center point, get pose size, normalize pose landmarks, landmarks to embedding, preprocess data
        person_numpy = normalize_person(person)

        # Check All Keypoints Visible
        # Check Keypoint Score is Above Threshold
        min_score = min([keypoint.score for keypoint in person.keypoints])

        color = (255, 255, 255)

        # Error Message
        if min_score < keypoint_detection_threshold_for_classifier:
            print("Make sure the person is fully visible in the camera.")

        # Pose Classification
        else:
            prob_list = classifier.classify_pose(person_numpy)
            # print(prob_list)

            top_class_name = prob_list[0].label
            # print(top_class_name)

            if top_class_name == "Average":
                color = (0, 155, 255)
            elif top_class_name == "Good":
                color = (0, 255, 255)
            elif top_class_name == "Bad":
                color = (0, 0, 255)
            elif top_class_name == "Perfect":
                color = (0, 255, 0)

        # Visualise video frames with person outline
        frame = utils.visualize(frame, list_persons, color)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        
        # Display frame
        cv2.imshow(estimation_model, frame)

        # Leave small gap between each frame to think and see if frame is classified correctly
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

# Selecting classifier, label file, movenet model and video to apply classification to
def main():
    classifier = "./TF-Models/Handstand/Handstand-Classifier.tflite"
    label_file = "./TF-Models/Handstand/Handstand-Labels.txt"
    estimation_model = "./TF-Movenet/movenet_lightning.tflite"
    video = "./ML-Test-Videos/angled-3.mp4"

    start = time.perf_counter()

    # Calling the run function to run classification process
    run(classifier, label_file, estimation_model, video)

    end = time.perf_counter()

    print(f'''
        ############################
        TOTAL TIME: {end - start:0.5f} seconds
        ############################
    ''')

if __name__ == '__main__':
  main()