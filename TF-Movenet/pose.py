import numpy as np
import cv2
import enum
from model import TensorFlowModel

class BodyPart(enum.Enum):
    """Enum representing human body keypoints detected by pose estimation models."""
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

class Pose:

    MIN_CROP_KEYPOINT_SCORE = 0.2
    TORSO_EXPANSION_RATIO = 1.9
    BODY_EXPANSION_RATIO = 1.2

    def __init__(self, model_name):

        model_interpreter = TensorFlowModel()
        model_interpreter.load(model_name, 4)

        self._model_interpreter = model_interpreter
        self.crop_region = None

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

            # If the body is large enough, there's no need to apply cropping logic.
            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            # Calculate the crop region that nicely covers the full body.
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

    def detect(self, input_image):
        
        image_height, image_width, _ = input_image.shape

        if (self.crop_region is None):
            self.crop_region = self.init_crop_region(image_height, image_width)

        keypoint_with_scores = self.run_detector(
            input_image,
            self.crop_region,
            crop_size=self._model_interpreter.get_crop_size()
        )

        self.crop_region = self.determine_crop_region(keypoint_with_scores,
                                                        image_height, image_width)

        return [keypoint_with_scores, image_width, image_height]