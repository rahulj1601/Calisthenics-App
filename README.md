# Calisthenics AI - Android Mobile App
### Rahul Jindal

---

## Description
What does this project do?

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
│   │   ├── requirements.txt
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
│   │   ├── Handstand
│   │   │   ├── Data
│   │   │   ├── Handstand-Classifier.tflite
│   │   │   ├── Handstand-Labels.txt
│   │   ├── Planche
│   │   │   ├── Data
│   │   │   ├── Planche-Classifier.tflite
│   │   │   ├── Planche-Labels.txt
│   │   ├── weights.best.hdf5
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

### Installing dependencies
*(Please ensure you have `Python 3.9` installed.)*\
*(This setup has been tested on DCS computers. Therefore, this would be ideal if you wish to run this project. A Linux machine is required for this process. )*

> Installing buildozer, Cython, virtualenv and python-for-android

```bash
pip3.9 install --user --upgrade pip
pip3.9 install --user --upgrade buildozer 
pip3.9 install --user --upgrade Cython==0.29.19 virtualenv 
pip3.9 install --user python-for-android
```

> Run the Follwing Commands to Create an `ml_venv` and Install requirements \
*(Please note you must be in the `Calisthenics-App` directory.)*

```bash
cd Calisthenics-App
python3.9 -m virtualenv ml_venv
source ml_venv/bin/activate
pip3.9 install -r requirements.txt
```

> Run the Follwing Commands to Create an `app_venv` and Install requirements\
*(Please note you must be in the `Calisthenics-App/App` directory.)*

<!-- Combine requirements.txt and test-requirements.txt files -->

```bash
cd Calisthenics-App/App
python3.9 -m virtualenv app_venv
source app_venv/bin/activate
pip3.9 install Cython==0.29.19
pip3.9 install -r requirements.txt
```

### Run Machine Learning TensorFlow Training Script

> Open the Jupyter Notebook called `train-ml-model.ipynb`. Set the kernel to `ml_venv`. Run the notebook by clicking "Run All".

<!-- TODO - Adjust Training Script - user selects variables for paths, delete certain directories before running. Remote model.py and pose.py from TF-Movenet if possible. -->

### Run Application on Computer

> Run the Application locally on your computer. \
*(Please note your computer will need a camera and certain features may not function exactly the same as they would on mobile.)*

```bash
cd Calisthenics-App/App
source app_venv/bin/activate
python main.py
```

### Run Application on Android

> Building & packaging the Android mobile application

```bash
cd Calisthenics-App/App
source app_venv/bin/activate
buildozer android debug
```

> Installing the APK File \
*(Please connect your Android mobile device into the computer.)*

```bash
adb install Calisthenics-App/App/bin/cali_ai-0.1-armeabi-v7a_arm64-v8a-debug.apk
```