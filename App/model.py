# Importing required modules
import numpy as np
from kivy.utils import platform

# Initialising TensorFlow Models for Android platform
if platform == 'android':
    # Import required Java classes using jnius library
    from jnius import autoclass

    # Define Java class names
    File = autoclass('java.io.File')
    Interpreter = autoclass('org.tensorflow.lite.Interpreter')
    InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
    Tensor = autoclass('org.tensorflow.lite.Tensor')
    DataType = autoclass('org.tensorflow.lite.DataType')
    TensorBuffer = autoclass('org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
    ByteBuffer = autoclass('java.nio.ByteBuffer')

    # Define TensorFlowModel class
    class TensorFlowModel():
        def load(self, model_filename, num_threads=None, label_file=None):
            # Load model file
            model = File(model_filename)

            # Create InterpreterOptions and set number of threads if provided
            options = InterpreterOptions()
            if num_threads is not None:
                options.setNumThreads(num_threads)

            # Create TensorFlow Lite Interpreter and allocate tensors
            self.interpreter = Interpreter(model, options)
            self.interpreter.allocateTensors()

            # Get input and output tensor details
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

            # If label_file is not provided, set input height and width from input tensor shape
            if label_file==None:
                self.input_height = self.input_shape[1]
                self.input_width = self.input_shape[2]

            # If label_file is provided, load the labels
            if label_file:
                self.pose_class_names = self.load_labels(label_file)

        def load_labels(self, label_path):
            # Load labels from file
            with open(label_path, 'r') as f:
                return [line.strip() for _, line in enumerate(f.readlines())]

        def get_crop_size(self):
            # Get input height and width as crop size
            return (self.input_height, self.input_width)

        def classify_pose(self, input_tensor):
            # Perform pose classification on input tensor
            output = self.pred(input_tensor)
            output = np.squeeze(output, axis=0)

            # Sort output by probability descending.
            prob_descending = sorted(range(len(output)), key=lambda k: output[k], reverse=True)
            prob_list = [
                [self.pose_class_names[idx], output[idx]]
                for idx in prob_descending
            ]

            return prob_list

        def pred(self, x):
            # Perform inference on input tensor and return output as NumPy array
            input = ByteBuffer.wrap(x.tobytes())
            output = TensorBuffer.createFixedSize(self.output_shape, self.output_type)
            self.interpreter.run(input, output.getBuffer().rewind())
            return np.reshape(np.array(output.getFloatArray()), self.output_shape)

# Initialising TensorFlow model for non Android platforms
else:
    import tensorflow as tf

    class TensorFlowModel:
        def load(self, model_filename, num_threads=None, label_file=None):
            # Load TensorFlow Lite model from file and allocate tensors
            self.interpreter = tf.lite.Interpreter(model_filename, num_threads=num_threads)
            self.interpreter.allocate_tensors()

            # Get input and output index of the interpreter
            self.input_index = self.interpreter.get_input_details()[0]['index']
            self.output_index = self.interpreter.get_output_details()[0]['index']

            # If label_file is not provided, get input height and width from the model
            if label_file == None:
                self.input_height = self.interpreter.get_input_details()[0]['shape'][1]
                self.input_width = self.interpreter.get_input_details()[0]['shape'][2]

            # If label_file is provided, load labels from the file
            if label_file:
                self.pose_class_names = self.load_labels(label_file)

        def load_labels(self, label_path):
            # Load labels from the provided label file
            with open(label_path, 'r') as f:
                return [line.strip() for _, line in enumerate(f.readlines())]

        def get_crop_size(self):
            # Return the input height and width as a tuple
            return (self.input_height, self.input_width)

        def classify_pose(self, input_tensor):
            # Classify input_tensor using the loaded model
            output = self.pred(input_tensor)
            output = np.squeeze(output, axis=0)

            # Sort output by probability descending.
            prob_descending = sorted(range(len(output)), key=lambda k: output[k], reverse=True)
            prob_list = [
                [self.pose_class_names[idx], output[idx]]
                for idx in prob_descending
            ]

            return prob_list

        def pred(self, x):
            # Set input tensor, invoke the interpreter, and get output tensor
            self.interpreter.set_tensor(self.input_index, x)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_index)
