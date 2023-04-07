import numpy as np
import cv2
import enum
from model import TensorFlowModel

# Enum to represent the key BodyParts
class BodyPart(enum.Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

# This class is used to isolate people from a frame and map out their body parts
# This is used by the movenet_thunder and movenet_lightning TensorFlow models
# The detected person is normalised within the frame to apply these TensorFlow models
class Pose:

    MIN_CROP_KEYPOINT_SCORE = 0.2
    TORSO_EXPANSION_RATIO = 1.9
    BODY_EXPANSION_RATIO = 1.2

    # Initialising the class and creating instance of TensorFlowModel to apply movenet_thunder/movenet_lightning
    def __init__(self, model_name):

        model_interpreter = TensorFlowModel()
        model_interpreter.load(model_name, 4)

        self._model_interpreter = model_interpreter
        self.crop_region = None

    # Initialising a crop region
    def init_crop_region(self, image_height, image_width):
        if image_width > image_height:
            x_min = 0.0
            box_width = 1.0
            y_min = (image_height / 2 - image_width / 2) / image_height
            box_height = image_width / image_height
        else:
            y_min = 0.0
            box_height = 1.0
            x_min = (image_width / 2 - image_height / 2) / image_width
            box_width = image_height / image_width

        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }

    # Isolating the torso from the image
    def torso_visible(self, keypoints):
        left_hip_score = keypoints[BodyPart.LEFT_HIP.value, 2]
        right_hip_score = keypoints[BodyPart.RIGHT_HIP.value, 2]
        left_shoulder_score = keypoints[BodyPart.LEFT_SHOULDER.value, 2]
        right_shoulder_score = keypoints[BodyPart.RIGHT_SHOULDER.value, 2]

        left_hip_visible = left_hip_score > Pose.MIN_CROP_KEYPOINT_SCORE
        right_hip_visible = right_hip_score > Pose.MIN_CROP_KEYPOINT_SCORE
        left_shoulder_visible = left_shoulder_score > Pose.MIN_CROP_KEYPOINT_SCORE
        right_shoulder_visible = right_shoulder_score > Pose.MIN_CROP_KEYPOINT_SCORE

        return ((left_hip_visible or right_hip_visible) and
                (left_shoulder_visible or right_shoulder_visible))

    # Determining torso sizes and distances from torso to key bodyparts
    def determine_torso_and_body_range(self, keypoints, target_keypoints, center_y, center_x):
        torso_joints = [ BodyPart.LEFT_SHOULDER, 
                         BodyPart.RIGHT_SHOULDER, 
                         BodyPart.LEFT_HIP, 
                         BodyPart.RIGHT_HIP 
                        ]
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for idx in range(len(BodyPart)):
            if keypoints[BodyPart(idx).value, 2] < Pose.MIN_CROP_KEYPOINT_SCORE:
                continue
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y

            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    # Figuring out area of image to crop and focus on
    def determine_crop_region(self, keypoints, image_height, image_width):
        target_keypoints = {}
        for idx in range(len(BodyPart)):
            target_keypoints[BodyPart(idx)] = [
                keypoints[idx, 0] *
                image_height, keypoints[idx, 1] * image_width
            ]

        if self.torso_visible(keypoints):
            center_y = (target_keypoints[BodyPart.LEFT_HIP][0] +
                        target_keypoints[BodyPart.RIGHT_HIP][0]) / 2
            center_x = (target_keypoints[BodyPart.LEFT_HIP][1] +
                        target_keypoints[BodyPart.RIGHT_HIP][1]) / 2

            (max_torso_yrange, max_torso_xrange, max_body_yrange,
             max_body_xrange) = self.determine_torso_and_body_range(
                 keypoints, target_keypoints, center_y, center_x)

            crop_length_half = np.amax([
                max_torso_xrange * Pose.TORSO_EXPANSION_RATIO,
                max_torso_yrange * Pose.TORSO_EXPANSION_RATIO,
                max_body_yrange * Pose.BODY_EXPANSION_RATIO,
                max_body_xrange * Pose.BODY_EXPANSION_RATIO
            ])

            distances_to_border = np.array(
                [center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin(
                [crop_length_half, np.amax(distances_to_border)])

            # Cropping is not applied if the person is a large fraction of the image
            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            
            # Calculate crop region to isolate whole person
            else:
                crop_length = crop_length_half * 2
            crop_corner = [center_y - crop_length_half,
                           center_x - crop_length_half]

            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            else:
                crop_length = crop_length_half * 2
            crop_corner = [center_y - crop_length_half,
                           center_x - crop_length_half]
            return {
                'y_min':
                    crop_corner[0] / image_height,
                'x_min':
                    crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                crop_corner[1] / image_width
            }
        else:
            return self.init_crop_region(image_height, image_width)

    # Crop the image using previously calculated dimensions
    def crop_and_resize(self, image, crop_region, crop_size):
        y_min, x_min, y_max, x_max = [
            crop_region['y_min'], crop_region['x_min'],
            crop_region['y_max'], crop_region['x_max']
        ]

        crop_top = int(0 if y_min < 0 else y_min * image.shape[0])
        crop_bottom = int(image.shape[0] if y_max >=
                          1 else y_max * image.shape[0])
        crop_left = int(0 if x_min < 0 else x_min * image.shape[1])
        crop_right = int(image.shape[1] if x_max >=
                         1 else x_max * image.shape[1])

        padding_top = int(0 - y_min * image.shape[0] if y_min < 0 else 0)
        padding_bottom = int((y_max - 1) * image.shape[0] if y_max >= 1 else 0)
        padding_left = int(0 - x_min * image.shape[1] if x_min < 0 else 0)
        padding_right = int((x_max - 1) * image.shape[1] if x_max >= 1 else 0)

        output_image = image[crop_top:crop_bottom, crop_left:crop_right]
        output_image = cv2.copyMakeBorder(output_image, padding_top, padding_bottom,
                                          padding_left, padding_right,
                                          cv2.BORDER_CONSTANT)
        output_image = cv2.resize(output_image, (crop_size[0], crop_size[1]))

        return output_image

    # Detect keypoints within crop region of image and also gather scores for different keypoints
    def run_detector(self, image, crop_region, crop_size):
        input_image = self.crop_and_resize( image, crop_region, crop_size=crop_size )
        input_image = input_image.astype(dtype=np.uint8)

        # Run model inference.
        keypoints_with_scores = self._model_interpreter.pred(np.expand_dims(input_image, axis=0))
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        # Update the coordinates.
        for idx in range(len(BodyPart)):
            keypoints_with_scores[idx, 0] = crop_region[
                'y_min'] + crop_region['height'] * keypoints_with_scores[idx, 0]
            keypoints_with_scores[idx, 1] = crop_region[
                'x_min'] + crop_region['width'] * keypoints_with_scores[idx, 1]

        return keypoints_with_scores

    # Obtain the center point of the left and right parts of the body
    def get_center_point(self, landmarks, left_bodypart, right_bodypart):
        left = landmarks[:, left_bodypart.value]
        right = landmarks[:, right_bodypart.value]
        center = left * 0.5 + right * 0.5
        return center

    # Calculate the largest possible size of the pose
    def get_pose_size(self, landmarks, torso_size_multiplier=2.5):
       
        # Hips center
        hips_center = self.get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

        # Shoulders center
        shoulders_center = self.get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)

        # Torso size as the minimum body size
        torso_size = np.linalg.norm(shoulders_center - hips_center)

        # Pose center
        pose_center_new = self.get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
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

    # Normalise the pose - move to center (0,0) and scale to constant pose size to match training set of poses
    def normalize_pose_landmarks(self, landmarks):
        
        # Move landmarks so that the pose center becomes (0,0)
        pose_center = self.get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
        pose_center = np.expand_dims(pose_center, axis=1)
        
        # Broadcast the pose center to the same size as the landmark vector to perform substraction
        pose_center = np.broadcast_to(pose_center, (landmarks.shape[0], 17, 2))
        landmarks = landmarks - pose_center

        # Scale the landmarks to a constant pose size
        pose_size = self.get_pose_size(landmarks)
        landmarks /= pose_size

        return landmarks

    # Convert the landmarks and scores array to a flattened numpy array embedding
    def landmarks_to_embedding(self, landmarks_and_scores):
        # Reshape the landmarks_and_scores array into a 17x3 array
        reshaped_inputs = np.reshape(landmarks_and_scores, (17, 3))

        # Extract the first 2 columns (x, y coordinates) from reshaped_inputs, normalize the landmarks using normalize_pose_landmarks() function, and store them in 'landmarks' variable
        landmarks = self.normalize_pose_landmarks(np.array([reshaped_inputs[:, :2]]))
            
        # Flatten the 'landmarks' array and store it in 'embedding' variable
        embedding = landmarks.flatten()
        
        # Return the flattened landmarks array as an embedding
        return embedding

    # Apply normalisation to the person in order to scale and position identical to training set
    def normalize_person(self, keypoints_with_scores):
        # Rearrange the x, y, score values from keypoints_with_scores array and store them as separate lists in 'landmarks' variable
        landmarks = [[x, y, score] for y, x, score in keypoints_with_scores]
        
        # Flatten the 'landmarks' list of lists and store it back in 'landmarks' variable as a single list
        landmarks = [item for sublist in landmarks for item in sublist]

        # Convert the 'landmarks' list into a numpy array and store it in 'landmarks_array' variable
        landmarks_array = np.array(landmarks)
        
        # Pass the 'landmarks_array' to landmarks_to_embedding() function to get an embedding, flatten it, and store it in 'embedding' variable
        embedding = self.landmarks_to_embedding(landmarks_array).reshape(34)

        # Flatten the 'embedding' array, convert it to float32 data type, and store it in 'input_tensor' variable
        input_tensor = np.array(embedding).flatten().astype(np.float32)
        
        # Add an extra dimension at the beginning of 'input_tensor' array along the first axis and store it back in 'input_tensor' variable
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Return the input tensor with normalized person keypoints
        return input_tensor
    
    # Detect the exact coordinates of body parts using movenet
    def detect(self, input_image):
        # Get the height, width, and number of channels of the input image and store them in respective variables
        image_height, image_width, _ = input_image.shape

        if (self.crop_region is None):
            # If the crop region is not initialized, then call the init_crop_region() function to initialize it with default values based on image height and width
            self.crop_region = self.init_crop_region(image_height, image_width)

        # Call the run_detector() function with input image, crop region, and crop size to get the keypoints with scores
        keypoint_with_scores = self.run_detector(
            input_image,
            self.crop_region,
            crop_size=self._model_interpreter.get_crop_size()
        )

        # Update the crop region using determine_crop_region() function based on the keypoints with scores, image height, and width
        self.crop_region = self.determine_crop_region(keypoint_with_scores,
                                                        image_height, image_width)

        # Call the normalize_person() function with keypoints with scores to get the normalized input tensor
        input_tensor = self.normalize_person(keypoint_with_scores)

        # Return a list containing input tensor, keypoints with scores, image width, and image height as outputs
        return [input_tensor, keypoint_with_scores, image_width, image_height]