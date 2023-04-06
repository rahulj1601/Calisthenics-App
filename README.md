# Calisthenics AI - Android Mobile App
### Rahul Jindal

---

## Description
What does this project do?
* Track your form for Handstand, Front Lever and Planche.
* Import previously recorded videos or record in real-time.
* Find out whether your form was Bad, Average, Good or Perfect.

---

## Directory Structure
*(These are the key folders & files within each sub-directory.)*

```markdown
├── Calisthenics-App
│   ├── App
│   │   ├── .buildozer
│   │   ├── bin
│   │   │   ├── cali_ai-0.1-armeabi-v7a_arm64-v8a-debug.apk
│   │   ├── camerax_provider
│   │   ├── data
│   │   ├── ML-Models
│   │   │   ├── Front Lever
│   │   │   ├── Handstand
│   │   │   ├── Planche
│   │   │   ├── movenet_lightning.tflite
│   │   │   ├── movenet_thunder.tflite
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── pose.py
│   │   ├── android_permissions.py
│   │   ├── buildozer.spec
│   │   ├── app_requirements.txt
│   │   ├── computer_requirements.txt
│   ├── Input
│   │   ├── Front Lever
│   │   ├── Handstand
│   │   ├── Planche
│   │   ├── Split-Front Lever
│   │   ├── Split-Handstand
│   │   ├── Split-Planche
│   ├── TF-Models
│   │   ├── Front Lever
│   │   │   ├── Data
│   │   │   ├── Front Lever-Classifier.tflite
│   │   │   ├── Front Lever-Labels.txt
│   │   │   ├── weights.best.hdf5
│   │   ├── Handstand
│   │   │   ├── Data
│   │   │   ├── Handstand-Classifier.tflite
│   │   │   ├── Handstand-Labels.txt
│   │   │   ├── weights.best.hdf5
│   │   ├── Planche
│   │   │   ├── Data
│   │   │   ├── Planche-Classifier.tflite
│   │   │   ├── Planche-Labels.txt
│   │   │   ├── weights.best.hdf5
│   ├── TF-Movenet
│   │   ├── examples
│   │   ├── movenet_lightning.tflite
│   │   ├── movenet_thunder.tflite
│   ├── requirements.txt
│   ├── label-clean-data.py
│   ├── test_ml_model.py
└────── train_ml_model.py
```

---

## Setup

### Creating Virtual Environments & Installing Dependencies
*(Please ensure you have `Python 3.9` installed.)*

> Installing buildozer, Cython, virtualenv and python-for-android

```bash
pip3.9 install --user --upgrade pip
pip3.9 install --user --upgrade buildozer 
pip3.9 install --user --upgrade Cython==0.29.19 virtualenv 
pip3.9 install --user python-for-android
```

> Run the Follwing Commands to Create an `ml_venv` and Install requirements \
*(This is required to run the `train-ml-model.py` file successfully.)*

```bash
cd Calisthenics-App
python3.9 -m virtualenv ml_venv
source ml_venv/bin/activate
pip3.9 install -r app_requirements.txt
```

> Run the Follwing Commands to Create an `computer_venv` and Install requirements\
*(This is required to run the application within your computer.)*

```bash
cd Calisthenics-App/App
python3.9 -m virtualenv computer_venv
source computer_venv/bin/activate
pip3.9 install Cython==0.29.19
pip3.9 install -r comp_requirements.txt
```

> Run the Follwing Commands to Create an `app_venv` and Install requirements\
*(This is required to build, package and run the Android mobile application.)*

```bash
cd Calisthenics-App/App
python3.9 -m virtualenv app_venv
source app_venv/bin/activate
pip3.9 install Cython==0.29.19
pip3.9 install -r app_requirements.txt
```

### Run TensorFlow ML Training Script
*(Please ensure you have successfully created the `ml_venv` as mentioned above.)*

1. Open the `train-ml-model.ipynb` Jupyter Notebook within VSCode. 
1. Set the kernel to `ml_venv`. 
1. Run the notebook by clicking "Run All".
1. Note: You may be prompted to install the ipykernel package within VSCode. Click "Install".

### Run Application on Computer
*(This should be possible with any computer including a video-camera, it has been tested to function correctly with a MacBook Pro Intel Core i5 Early 2015.)*

```bash
cd Calisthenics-App/App
source computer_venv/bin/activate
python main.py
```

### Run Application on Android
*(The packaging and building of the Android mobile application has been tested on Warwick DCS computers. Theoretically, any Linux computer should suffice. The APK can be installed from any computer onto any Android mobile device. Please note, the `buildozer android debug` command can take a while to run up to 1hr. You can skip the building and packaging steps if you wish, and go straight to installing the APK onto your Android device.)*

> Building & packaging the Android mobile application

```bash
cd Calisthenics-App/App
source app_venv/bin/activate
buildozer android clean
buildozer android debug
```

> Installing the APK File

1. Install `adb` on your computer if it is not already installed.
1. Enable USB debugging on your Android mobile device.
1. Connect your Android mobile device to the computer.
1. Finally, run the following command:
```bash
adb install Calisthenics-App/App/bin/cali_ai-0.1-armeabi-v7a_arm64-v8a-debug.apk
```