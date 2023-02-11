from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.core.window import Window
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.clock import mainthread
from kivy.graphics import Color, Rectangle
import cv2
import numpy as np
import os
import datetime

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDFillRoundFlatIconButton

from camera4kivy import Preview

from pose import Pose
from model import TensorFlowModel
from android_permissions import AndroidPermissions

Builder.load_string("""
<AppLayout>:
    video_preview: self.ids.camera
    exercise_form_widget: self.ids.exercise_form
    recording_widget: self.ids.recording
    bottom_bar_widget: self.ids.bottom_bar

    VideoPreview:
        id: camera
        orientation: 'same'

    MDBottomAppBar:
        MDTopAppBar:
            id: bottom_bar
            title: "Calisthenics AI"
            icon: "video"
            on_action_button: app.toggle_video()
            icon_color: (1, 1, 1, 1)
            mode: "free-center"
            type: "bottom"
            left_action_items: [["camera-flip-outline", lambda x: app.switch_camera()]]
            right_action_items: [["gymnastics", lambda x: app._camera.handstand()], ["weight-lifter", lambda x: app._camera.muscleup()], ["dumbbell", lambda x: app._camera.planche()]]

    RelativeLayout:
        MDFillRoundFlatIconButton:
            id: recording
            icon: "circle"
            text: "00:00"
            text_color: "white"
            md_bg_color: (0,0,0,1)
            icon_color: (1, 1, 1, 1)
            pos_hint: {'right': 0.99, 'top': 0.99}

    RelativeLayout:
        MDFillRoundFlatIconButton:
            id: exercise_form
            text: "Handstand"
            icon: "gymnastics"
            text_color: "white"
            icon_color: (1, 1, 1, 1)
            pos_hint: {'left': 0.99, 'top': 0.99}
""")

class AppLayout(FloatLayout):
    video_preview = ObjectProperty()
    exercise_form_widget = ObjectProperty()
    recording_widget = ObjectProperty()
    bottom_bar_widget = ObjectProperty()

class VideoPreview(Preview):
    # Load Movenet Model (Lightning or Thunder)
    pose_detector = Pose(os.path.join(os.getcwd(), 'ML-Models/movenet_lightning.tflite'))

    # Score Threshold Per Keypoint
    keypoint_detection_threshold_for_classifier = 0.1

    # Classification Model
    classifier = TensorFlowModel()
    classifier.load(os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Classifier.tflite'), 
                    4, 
                    os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Labels.txt'))

    current_pose = "Handstand"
    icons = {
        "Handstand": "gymnastics",
        "Muscle Up": "weight-lifter",
        "Planche": "dumbbell"
    }

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
        super().__init__(**kwargs)
        self.analyzed_texture = None
        self.app = App.get_running_app()

    def analyze_pixels_callback(self, pixels, image_size, image_pos, scale, mirror):
        frame = np.frombuffer(pixels, np.uint8).reshape(image_size[1], image_size[0], 4)
        frame = frame[:,:,:3]

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

        self.make_thread_safe(frame.tobytes(), (frame.shape[1], frame.shape[0]))

    @mainthread
    def make_thread_safe(self, pixels, size):
        if not self.analyzed_texture or\
           self.analyzed_texture.size[0] != size[0] or\
           self.analyzed_texture.size[1] != size[1]:
            self.analyzed_texture = Texture.create(size=size, colorfmt='rgb')
            self.analyzed_texture.flip_vertical()
        self.analyzed_texture.blit_buffer(pixels, colorfmt='rgb', bufferfmt='ubyte') 

    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        if self.analyzed_texture:
            Rectangle(texture = self.analyzed_texture,
                      size = Window.size, 
                      pos = self.pos)

    def process_frame(self, frame):
        
        # Flip Frame (Recorded with Front Facing Camera)
        #frame = cv2.flip(frame, 1)

        # Isolate People from Frame
        person, image_width, image_height = self.pose_detector.detect(frame)

        # Check Keypoints are Visible
        min_score = min([z for x,y,z in person])

        colour = (0,0,0)

        # Error Message
        if min_score < self.keypoint_detection_threshold_for_classifier:
            colour = (0,0,0)
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
                colour = (255,155,0)
            elif top_class_name == "Good":
                colour = (0,255,0)
            elif top_class_name == "Bad":
                colour = (255,0,0)

            frame = self.visualize(frame, person, colour, image_width, image_height)

        self.app._exercise_form.text = self.current_pose
        self.app._exercise_form.icon = self.icons[self.current_pose]
        self.app._exercise_form.md_bg_color = (colour[0]/255, colour[1]/255, colour[2]/255, 1)

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

    def handstand(self):
        self.current_pose = "Handstand"
        self.classifier.load(os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Classifier.tflite'), 
                    4, 
                    os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Labels.txt'))

    def muscleup(self):
        self.current_pose = "Muscle Up"
        self.classifier.load(os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Classifier.tflite'), 
                    4, 
                    os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Labels.txt'))

    def planche(self):
        self.current_pose = "Planche"
        self.classifier.load(os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Classifier.tflite'), 
                    4, 
                    os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Labels.txt'))

class MyApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"
        self.theme_cls.material_style = "M2"
        self.title = "Calisthenics AI Personal Trainer"

        self.layout = AppLayout()
        self._camera = self.layout.video_preview
        self._exercise_form = self.layout.exercise_form_widget
        self._recording = self.layout.recording_widget
        self._bottom_bar = self.layout.bottom_bar_widget

        self.recording = False

        return self.layout

    def on_start(self):
        self.dont_gc = AndroidPermissions(self.start_app)

    def start_app(self):
        self.dont_gc = None
        Clock.schedule_once(self.connect_camera)
        '''if platform == 'android':
            self._camera.zoom_delta(delta_scale = 0.5)'''

    def connect_camera(self, dt):
        self._camera.connect_camera(analyze_pixels_resolution = 720, 
                                   enable_analyze_pixels = True)

    def on_stop(self):
        self._camera.disconnect_camera()

    def toggle_video(self):
        if self.recording:
            self._camera.stop_capture_video()
            self._recording.md_bg_color = (0, 0, 0, 1)
            self._recording.icon_color = (1, 1, 1, 1)
            self._recording.text_color = "white"
            self._bottom_bar.icon_color = (1, 1, 1, 1)
            self.event.cancel()
            self._recording.text = "00:00"
            self.recording = False
        else:
            self._camera.capture_video()
            self._recording.md_bg_color = (1, 1, 1, 1)
            self._recording.icon_color = (1, 0, 0, 1)
            self._recording.text_color = "red"
            self._bottom_bar.icon_color = (1, 0, 0, 1)
            self.event = Clock.schedule_interval(self.update_time, 1)
            self.recording = True
    
    def update_time(self, dt):
        minutes = int(self._recording.text[0:2])
        seconds = int(self._recording.text[3:5])
        a = datetime.datetime(100,1,1,11,minutes,seconds)
        b = a + datetime.timedelta(seconds = 1)
        self._recording.text = f"{b.minute:02}" + ":" + f"{b.second:02}"

    def switch_camera(self):
        self._camera.select_camera('toggle')
    
if __name__ == '__main__':
    MyApp().run()