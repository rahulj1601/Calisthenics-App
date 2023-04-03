# Calisthenics ML Android App
### Rahul Jindal

---

### Description
What does this project do?

---

### Directory Structure
*(These are the key folders & files within each sub-directory.)*

```markdown
├── Calisthenics-App
│   ├── App
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
│   ├── Input
│   │   ├── Front Lever
│   │   │   ├── Bad
│   │   │   ├── Average
│   │   │   ├── Good
│   │   │   ├── Perfect
│   │   ├── Handstand
│   │   │   ├── Bad
│   │   │   ├── Average
│   │   │   ├── Good
│   │   │   ├── Perfect
│   │   ├── Planche
│   │   │   ├── Bad
│   │   │   ├── Average
│   │   │   ├── Good
│   │   │   ├── Perfect
│   │   ├── Split-Front Lever
│   │   │   ├── test
│   │   │   ├── train
│   │   ├── Split-Handstand
│   │   │   ├── test
│   │   │   ├── train
│   │   ├── Split-Planche
│   │   │   ├── test
│   │   │   ├── train
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
│   ├── label-clean-data.py
│   ├── test_ml_model.py
└────── train_ml_model.py
```

---

### Setup
How to setup this project and try it out.

```bash
java -cp lib/gson-2.10.1.jar:src RoundupApp
```

*(NOTE: The project should already be compiled.)*

---