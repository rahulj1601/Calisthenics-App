import numpy as np
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics import Rectangle, Color
from kivy.uix.widget import Widget

from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDFloatingActionButton

from kivy.lang import Builder

import cv2

from pose import Pose
from model import TensorFlowModel

class VideoWidget(BoxLayout):

    # Load Movenet Model (Lightning or Thunder)
    pose_detector = Pose("ML-Models/movenet_lightning.tflite")

    # Video Capture
    cap = cv2.VideoCapture(0)

    # Score Threshold Per Keypoint
    keypoint_detection_threshold_for_classifier = 0.0

    # Classification Model
    classifier = TensorFlowModel()
    classifier.load("ML-Models/Handstand/Handstand-Classifier.tflite", 4, "ML-Models/Handstand/Handstand-Labels.tflite")

    edge_pairs = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16)
    ]

    def __init__(self, **kwargs):
        super(VideoWidget, self).__init__(**kwargs)

        self.capture = cv2.VideoCapture(0)
 
        # Arranging Canvas
        with self.canvas:
 
            #Color(.234, .456, .678, .8)  # set the colour
 
            # Setting the size and position of canvas
            self.rect = Rectangle(pos = self.pos,
                                  size = Window.size)
 
            # Update the canvas as the screen size change
            self.bind(pos = self.update_rect, size = self.update_rect)

        Clock.schedule_interval(self.update_rect, 1.0/33.0)
 
    # update function which makes the canvas adjustable.
    def update_rect(self, *args):
        self.rect.size = Window.size
        self.rect.pos = self.pos

        # display image from cam in opencv window
        
        ret, frame = self.capture.read()

        # Setting the frame to the window size
        window_w, window_h = Window.size
        frame_h, frame_w, _ = frame.shape

        scale_factor = max([window_w/frame_w, window_h/frame_h])

        frame_w, frame_h = int(frame_w*scale_factor), int(frame_h*scale_factor)
        midpoint_w, midpoint_h = int(frame_w/2), int(frame_h/2)
        frame = cv2.resize(frame, (frame_w, frame_h))

        frame = frame[int(midpoint_h-(window_h/2)) : int(midpoint_h+(window_h/2)), 
                      int(midpoint_w-(window_w/2)) : int(midpoint_w+(window_w/2))]

        frame = self.process_frame(frame)

        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        # display image from the texture
        self.rect.texture = texture1

    def process_frame(self, frame):
        
        # Flip Frame (Recorded with Front Facing Camera)
        frame = cv2.flip(frame, 1)

        # Isolate People from Frame
        person, image_width, image_height = self.pose_detector.detect(frame)

        # Check Keypoints are Visible
        min_score = min([z for x,y,z in person])

        colour = (255,255,255)

        # Error Message
        if min_score < self.keypoint_detection_threshold_for_classifier:
            print("Make sure the person is fully visible in the camera.")
        
        # Pose Classification
        else:
            input_tensor = [[
                y,x,score
            ] for y,x,score in person]
            input_tensor = np.array(input_tensor).flatten().astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            prob_list = self.classifier.classify_pose(input_tensor)
            top_class_name = prob_list[0][0]

            if top_class_name == "Average":
                colour = (0,155,255)
            elif top_class_name == "Good":
                colour = (0,255,0)
            elif top_class_name == "Bad":
                colour = (0,0,255)

        frame = self.visualize(frame, person, colour, image_width, image_height)

        return frame

    def visualize(self, image, keypoints, colour, image_width, image_height):
        keypoint_threshold = 0.05
        
        # Draw all the landmarks
        for i in range(len(keypoints)):
            if keypoints[i][2] >= keypoint_threshold:
                cv2.circle(image, (int(keypoints[i][1]*image_width), int(keypoints[i][0]*image_height)), 2, colour, 4)

        # Draw all the edges
        for edge_pair in self.edge_pairs:
            if (keypoints[edge_pair[0]][2] > keypoint_threshold and
                    keypoints[edge_pair[1]][2] > keypoint_threshold):
                cv2.line(image, (int(keypoints[edge_pair[0]][1]*image_width), int(keypoints[edge_pair[0]][0]*image_height)),
                         (int(keypoints[edge_pair[1]][1]*image_width), int(keypoints[edge_pair[1]][0]*image_height)), colour, 2)
        
        return image

KV = """
MDScreen:
    MDBoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            title: "Calisthenics AI"
            right_action_items: [["dots-vertical", lambda x: app.callback_1()], ["clock", lambda x: app.callback_2()]]

"""

class MyApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"
        self.title = "Calisthenics AI Personal Trainer"

        widget = Builder.load_string(KV)

        screen = VideoWidget()

        screen.add_widget(widget)

        return screen

if __name__ == '__main__':
    MyApp().run()
    cv2.destroyAllWindows()