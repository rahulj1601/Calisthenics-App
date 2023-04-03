from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.core.window import Window
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.clock import mainthread
from kivy.graphics import Color, Rectangle
from kivy.properties import DictProperty
import cv2
import numpy as np
import os
import datetime

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.label import MDIcon
from kivymd.uix.screen import Screen
from kivymd.uix.screenmanager import ScreenManager
from kivy.uix.screenmanager import SlideTransition
from kivymd.uix.button import MDFillRoundFlatIconButton, MDFlatButton
from kivymd.uix.filemanager import MDFileManager
#from kivymd.uix.snackbar import BaseSnackbar
from kivymd.toast import toast 
from kivy.metrics import dp
from kivy.animation import Animation

from camera4kivy import Preview

from pose import Pose
from model import TensorFlowModel
from android_permissions import AndroidPermissions

if platform == "android":
    from android.storage import primary_external_storage_path

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
            icon: "video"
            on_action_button: app.toggle_video()
            md_bg_bottom_color: 0, 0, 0, 1
            icon_color: 0, 0, 0, 1
            opposite_colors: False
            mode: "center"
            type: "bottom"
            left_action_items: [["home-outline", lambda x: app.changeScreen('intro_layout')],["folder-outline", lambda x: app.changeScreen('file_layout')],["camera-flip-outline", lambda x: app.switch_camera()]]
            right_action_items: [["data/white-handstand.png", lambda x: app._camera.changePose('Handstand')], ["data/white-front-lever.png", lambda x: app._camera.changePose('Front Lever')], ["data/white-planche.png", lambda x: app._camera.changePose('Planche')]]

    RelativeLayout:
        MDFillRoundFlatIconButton:
            id: recording
            icon: "circle"
            text: "00:00"
            text_color: "white"
            md_bg_color: (0,0,0,1)
            icon_color: (1, 1, 1, 1)
            pos_hint: {"center_x": .80, "center_y": .95}

        MDFillRoundFlatIconButton:
            id: exercise_form
            text: "Handstand"
            icon_color: (1, 1, 1, 1)
            text_color: "white"
            icon: "alert-circle"
            pos_hint: {"center_x": .30, "center_y": .95}
            md_bg_color: 0, 0, 0, 1

<IntroLayout>:

    BoxLayout:

        orientation: 'vertical'

        Image:
            source: "data/logo-icon.png"
            size_hint_x: 0.4
            allow_stretch: True
            pos_hint: {"center_y": .7, "center_x": .5}
            
        MDScrollView:
            MDList:
                id: container

                pos_hint: {"center_y": .5, "center_x": .5}

                TwoLineAvatarIconListItem:
                    text: "Get Started"
                    secondary_text: "Scroll to learn what each icon does."
                    IconRightWidget:
                        icon: ""

                OneLineAvatarIconListItem:
                    text: "Return to Main Menu"
                    IconRightWidget:
                        icon: "home-outline"

                OneLineAvatarIconListItem:
                    text: "Open Video/Image File Analyser"
                    IconRightWidget:
                        icon: "folder-outline"
                
                OneLineAvatarIconListItem:
                    text: "Open Realtime Video Analyser"
                    IconRightWidget:
                        icon: "camera-outline"

                OneLineAvatarIconListItem:
                    text: "Switch Camera"
                    IconRightWidget:
                        icon: "camera-flip-outline"

                OneLineAvatarIconListItem:
                    text: "Start/End Video Recording"
                    IconRightWidget:
                        icon: "video"
                
                OneLineAvatarIconListItem:
                    text: "Play Video"
                    IconRightWidget:
                        icon: "play"
                
                OneLineAvatarIconListItem:
                    text: "Rotate Video"
                    IconRightWidget:
                        icon: "file-rotate-right"

                TwoLineAvatarIconListItem:
                    text: "Handstand"
                    secondary_text: "Use side on view for best results"
                    IconRightWidget:
                        icon: "data/white-handstand.png"
                
                TwoLineAvatarIconListItem:
                    text: "Front Lever"
                    secondary_text: "Use side on view for best results"
                    IconRightWidget:
                        icon: "data/white-front-lever.png"

                TwoLineAvatarIconListItem:
                    text: "Planche"
                    secondary_text: "Use side on view for best results"
                    IconRightWidget:
                        icon: "data/white-planche.png"

                TwoLineAvatarIconListItem:
                    text: "Colour List"
                    secondary_text: "Learn what each colour represents."
                    IconRightWidget:
                        icon: ""

                OneLineAvatarIconListItem:
                    text: "Bad Form"
                    IconRightWidget:
                        icon: "data/red.png"
                
                OneLineAvatarIconListItem:
                    text: "Average Form"
                    IconRightWidget:
                        icon: "data/orange.png"

                OneLineAvatarIconListItem:
                    text: "Good Form"
                    IconRightWidget:
                        icon: "data/yellow.png"
                
                OneLineAvatarIconListItem:
                    text: "Perfect Form"
                    IconRightWidget:
                        icon: "data/green.png"

        RelativeLayout:

            MDLabel:
                text: "Choose a Button Below to Get Started!"
                halign: 'center'
                pos_hint: {"center_y": .8, "center_x": .5}

            MDFillRoundFlatIconButton:
                text: "Realtime Analyser"
                pos_hint: {"center_y": .5, "center_x": .7}
                on_release: app.changeScreen('app_layout')
                icon: "camera-outline"

            MDFillRoundFlatIconButton:
                text: "File Analyser"
                pos_hint: {"center_y": .5, "center_x": .3}
                on_release: app.changeScreen('file_layout')
                icon: "folder-outline"

<FileLayout>:
    file_video_preview: self.ids.preview
    file_form_widget: self.ids.form

    FileVideoPreview:
        id: preview

    MDBottomAppBar:
        MDTopAppBar:
            icon: "play-outline"
            on_action_button: app._file_video_preview.clicked_play()
            type: "bottom"
            mode: "center"
            opposite_colors: True
            md_bg_bottom_color: 1, 1, 1, 1
            icon_color: 0, 0, 0, 1
            right_action_items: [["data/black-handstand.png", lambda x: app._file_video_preview.changePose('Handstand')], ["data/black-front-lever.png", lambda x: app._file_video_preview.changePose('Front Lever')], ["data/black-planche.png", lambda x: app._file_video_preview.changePose('Planche')]]
            left_action_items: [["home", lambda x: app.changeScreen('intro_layout')], ["camera-outline", lambda x: app.changeScreen('app_layout')], ["file-rotate-right", lambda x: app._file_video_preview.rotateFrame()]]

    RelativeLayout:

        MDFillRoundFlatIconButton:
            text: "Open Video"
            icon: "folder-open-outline"
            pos_hint: {"center_x": .80, "center_y": .95}
            on_release: app.file_manager_open()
            text_color: "black"
            md_bg_color: 1, 1, 1, 1
            icon_color: 0, 0, 0, 1

        MDFillRoundFlatIconButton:
            id: form
            md_bg_color: 0, 0, 0, 1
            text: "No Person Detected"
            icon: "alert-circle"
            pos_hint: {"center_x": .30, "center_y": .95}
""")

class AppLayout(Screen):
    video_preview = ObjectProperty()
    exercise_form_widget = ObjectProperty()
    recording_widget = ObjectProperty()
    bottom_bar_widget = ObjectProperty()

class IntroLayout(Screen):
    pass

class FileLayout(Screen):
    file_video_preview = ObjectProperty()
    file_form_widget = ObjectProperty()

class PoseAnalysis:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load Movenet Model (Lightning or Thunder)
        self.pose_detector = Pose(os.path.join(os.getcwd(), 'ML-Models/movenet_lightning.tflite'))

        # Score Threshold Per Keypoint
        self.keypoint_detection_threshold_for_classifier = 0.1

        # Classification Model
        self.classifier = TensorFlowModel()
        self.classifier.load(os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Classifier.tflite'), 
                        4, 
                        os.path.join(os.getcwd(), 'ML-Models/Handstand/Handstand-Labels.txt'))

        self.edge_pairs = [
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

        self.colour_dict = {
            "Perfect": (0,255,0),
            "Good": (255,255,0),
            "Average": (255,155,0),
            "Bad": (255,0,0)
        }

    def process_frame(self, frame, flip=False):
        
        # Flip Frame (Recorded with Front Facing Camera)
        if flip:
            frame = cv2.flip(frame, 1)

        # Isolate People from Frame
        input_tensor, person, image_width, image_height = self.pose_detector.detect(frame)

        # Check Keypoints are Visible
        min_score = min([z for x,y,z in person])

        colour = (0,0,0)

        # Error Message
        if min_score < self.keypoint_detection_threshold_for_classifier:
            return frame, None
        
        # Pose Classification
        else:
            '''input_tensor = [[
                y,x,score
            ] for y,x,score in person]
            input_tensor = np.array(input_tensor).flatten().astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0)'''

            prob_list = self.classifier.classify_pose(input_tensor)
            top_class_name = prob_list[0][0]
            print(top_class_name)

            colour = self.colour_dict[top_class_name]

            frame = self.visualize(frame, person, colour, image_width, image_height)

        return frame, colour

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

class FileVideoPreview(Image):

    current_pose = "Handstand"

    icons = {
        "Handstand": "handstand.png",
        "Front Lever": "front-lever.png",
        "Planche": "planche.png"
    }

    pose_analyser = PoseAnalysis()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_file = None
        self.app = App.get_running_app()
        self.rotate = [3, "None"]
        self.texture = Texture.create(size=Window.size, colorfmt='rgb')
    
    def clicked_play(self):
        if self.current_file != None:
            self.start(self.current_file, 60)
    
    def start(self, file_path, fps):
        # Check if file is mp4
        if str(file_path[-4:]) == ".mp4" or str(file_path[-4:]) == ".mov" or str(file_path[-4:]) == ".gif" or str(file_path[-4:]) == ".jpg":
            self.current_file = file_path
            self.capture = cv2.VideoCapture(file_path)
            self.event = Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            if self.rotate[1] != "None":
                frame = cv2.rotate(frame, self.rotate[1])
            frame, colour = self.pose_analyser.process_frame(frame)

            if colour == None:
                colour = (0,0,0)
                self.app._file_form.text = "No Person Detected"
                self.app._file_form.icon = 'alert-circle'
                self.app._file_form.md_bg_color = (1, 0, 0, 1)
                self.app._file_form.pos_hint = {"center_x": .30, "center_y": .95}
            else:
                self.app._file_form.text = self.current_pose
                self.app._file_form.icon = "data/white-" + self.icons[self.current_pose]
                self.app._file_form.md_bg_color = (colour[0]/255, colour[1]/255, colour[2]/255, 1)
                self.app._file_form.pos_hint = {"center_x": .20, "center_y": .95}

            buf = cv2.flip(frame, 0).tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            
            self.texture = image_texture

        if self.app.layout.manager.current != "file_layout":
            Clock.unschedule(self.event)
    
    def changePose(self, pose):
        self.current_pose = pose
        self.pose_analyser.classifier.load(os.path.join(os.getcwd(), 'ML-Models/' + pose + '/' + pose + '-Classifier.tflite'), 
                    4, 
                    os.path.join(os.getcwd(), 'ML-Models/' + pose + '/' + pose + '-Labels.txt'))
    
    def rotateFrame(self):
        rotateDict = {
            0: cv2.ROTATE_90_CLOCKWISE,
            1: cv2.ROTATE_180,
            2: cv2.ROTATE_90_COUNTERCLOCKWISE,
            3: "None"
        }
        key = (self.rotate[0]+1) % 4
        self.rotate = [ key, rotateDict[key] ]

class VideoPreview(Preview):
    current_pose = "Handstand"

    icons = {
        "Handstand": "handstand.png",
        "Front Lever": "front-lever.png",
        "Planche": "planche.png"
    }

    pose_analyser = PoseAnalysis()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzed_texture = None
        self.app = App.get_running_app()
        self.flip = False

        self.started_writer = False
        self.recording = False
        self.out = None

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

        if self.recording and not self.started_writer and False:
            date = datetime.datetime.today().strftime("%Y_%m_%d")
            time = datetime.datetime.now().strftime("%H_%M_%S.%f")
            file_path = primary_external_storage_path() + '/DCIM/Calisthenics AI' + "/" + date + "/" + time + ".mp4"
            self.out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_w, frame_h))
            self.started_writer = True

        frame = frame[int(midpoint_h-(window_h/2)) : int(midpoint_h+(window_h/2)), 
                    int(midpoint_w-(window_w/2)) : int(midpoint_w+(window_w/2))]

        frame, colour = self.pose_analyser.process_frame(frame, self.flip)

        if colour == None:
            #Clock.schedule_once(lambda dt: self.showWarning(), 0)
            self.app._exercise_form.text = "No Person Detected"
            self.app._exercise_form.icon = 'alert-circle'
            self.app._exercise_form.md_bg_color = (1, 0, 0, 1)
            self.app._exercise_form.pos_hint = {"center_x": .30, "center_y": .95}
        else:
            self.app._exercise_form.text = self.current_pose
            self.app._exercise_form.icon = "data/white-" + self.icons[self.current_pose]
            self.app._exercise_form.md_bg_color = (colour[0]/255, colour[1]/255, colour[2]/255, 1)
            self.app._exercise_form.pos_hint = {"center_x": .20, "center_y": .95}

        if self.recording and self.started_writer and False:
            self.out.write(frame)

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

    def changePose(self, pose):
        self.current_pose = pose
        self.pose_analyser.classifier.load(os.path.join(os.getcwd(), 'ML-Models/' + pose + '/' + pose + '-Classifier.tflite'), 
                    4, 
                    os.path.join(os.getcwd(), 'ML-Models/' + pose + '/' + pose + '-Labels.txt'))
    
    def showWarning(self):
        '''snackbar = CustomSnackbar(
            text="Person Not in Frame",
            icon="information",
            snackbar_x="10dp",
            snackbar_y="10dp",
            pos_hint = {"center_x": .5, "center_y": .94}
        )
        snackbar.size_hint_x = (
            Window.width - (snackbar.snackbar_x * 2)
        ) / Window.width
        snackbar.open()'''

class MyApp(MDApp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager, select_path=self.select_path
        )

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.material_style = "M2"
        self.title = "Calisthenics AI Personal Trainer"

        self.layout = AppLayout(name = "app_layout")
        self._camera = self.layout.video_preview
        self._exercise_form = self.layout.exercise_form_widget
        self._recording = self.layout.recording_widget
        self._bottom_bar = self.layout.bottom_bar_widget

        self.recording = False

        self.file_manager_layout = FileLayout(name = "file_layout")
        self._file_video_preview = self.file_manager_layout.file_video_preview
        self._file_form = self.file_manager_layout.file_form_widget

        sm = ScreenManager(transition = SlideTransition())
        sm.add_widget(IntroLayout(name = "intro_layout"))
        sm.add_widget(self.layout)
        sm.add_widget(self.file_manager_layout)

        return sm

    def on_start(self):
        self.dont_gc = AndroidPermissions(self.start_app)

    def start_app(self):
        self.dont_gc = None
    
    def changeScreen(self, screen_name):
        if screen_name == "app_layout":
            Clock.schedule_once(self.connect_camera)
            self.layout.manager.transition.direction = "left"
            self._camera.flip = False

        elif screen_name == "intro_layout":
            if self.layout.manager.current == "app_layout":
                self._camera.disconnect_camera()
                self.layout.manager.transition.direction = "right"
            else:
                self.layout.manager.transition.direction = "left"

        elif screen_name == "file_layout":
            if self.layout.manager.current == "app_layout":
                self._camera.disconnect_camera()
            self.layout.manager.transition.direction = "right"

        self.layout.manager.current = screen_name

    def connect_camera(self, dt):
        self._camera.connect_camera(analyze_pixels_resolution = 640, 
                                   enable_analyze_pixels = True,
                                   default_zoom = 0.0)

    def on_stop(self):
        self._camera.disconnect_camera()

    def toggle_video(self):
        if self.recording:
            self._camera.stop_capture_video()
            self._recording.md_bg_color = (0, 0, 0, 1)
            self._recording.icon_color = (1, 1, 1, 1)
            self._recording.text_color = "white"
            self._bottom_bar.icon_color = (0, 0, 0, 1)
            Clock.unschedule(self.event)
            self._recording.text = "00:00"
            self.recording = False
            self._camera.recording = False
            self._camera.started_writer = False
            #self._camera.out.release()
        else:
            self._camera.capture_video()
            self._recording.md_bg_color = (1, 1, 1, 1)
            self._recording.icon_color = (1, 0, 0, 1)
            self._recording.text_color = "red"
            self._bottom_bar.icon_color = (1, 0, 0, 1)
            self.event = Clock.schedule_interval(self.update_time, 1)
            self.recording = True
            self._camera.recording = True
    
    def update_time(self, dt):
        minutes = int(self._recording.text[0:2])
        seconds = int(self._recording.text[3:5])
        a = datetime.datetime(100,1,1,11,minutes,seconds)
        b = a + datetime.timedelta(seconds = 1)
        self._recording.text = f"{b.minute:02}" + ":" + f"{b.second:02}"

        if self.layout.manager.current != "app_layout":
            self.toggle_video()

    def switch_camera(self):
        self._camera.select_camera('toggle')
        self._camera.flip=False if self._camera.flip else True
    
    ##### File Manager Functions

    def file_manager_open(self):
        file_path = os.path.expanduser("~")
        if platform == "android":
            file_path = primary_external_storage_path() + "/DCIM"
        self.file_manager.show(file_path)  # output manager to the screen
        self.manager_open = True

    def select_path(self, path: str):
        '''
        It will be called when you click on the file name
        or the catalog selection button.

        :param path: path to the selected directory or file;
        '''
        
        self._file_video_preview.start(path, 60)

        self.exit_manager()
        toast(path)

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True
    
if __name__ == '__main__':
    MyApp().run()