import numpy as np
from kivy.utils import platform

if platform == 'android':
    from jnius import autoclass

    File = autoclass('java.io.File')
    Interpreter = autoclass('org.tensorflow.lite.Interpreter')
    InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
    Tensor = autoclass('org.tensorflow.lite.Tensor')
    DataType = autoclass('org.tensorflow.lite.DataType')
    TensorBuffer = autoclass(
        'org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
    ByteBuffer = autoclass('java.nio.ByteBuffer')

    class TensorFlowModel():
        def load(self, model_filename, num_threads=None, label_file=None):
            model = File(model_filename)
            options = InterpreterOptions()
            if num_threads is not None:
                options.setNumThreads(num_threads)
            self.interpreter = Interpreter(model, options)
            self.interpreter.allocateTensors()

            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

            if label_file==None:
                self.input_height = self.input_shape[1]
                self.input_width = self.input_shape[2]

            if label_file:
                self.pose_class_names = self.load_labels(label_file)

        def load_labels(self, label_path):
            with open(label_path, 'r') as f:
                return [line.strip() for _, line in enumerate(f.readlines())]

        def get_crop_size(self):
            return (self.input_height, self.input_width)

        def classify_pose(self, input_tensor):
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
            input = ByteBuffer.wrap(x.tobytes())
            output = TensorBuffer.createFixedSize(self.output_shape, self.output_type)
            self.interpreter.run(input, output.getBuffer().rewind())
            return np.reshape(np.array(output.getFloatArray()), self.output_shape)

elif platform == 'ios':
    from pyobjus import autoclass, objc_arr
    from ctypes import c_float, cast, POINTER

    NSString = autoclass('NSString')
    NSError = autoclass('NSError')
    Interpreter = autoclass('TFLInterpreter')
    InterpreterOptions = autoclass('TFLInterpreterOptions')
    NSData = autoclass('NSData')
    NSMutableArray = autoclass("NSMutableArray")

    class TensorFlowModel:
        def load(self, model_filename, num_threads=None, label_file=None):

            self.error = NSError.alloc()
            model = NSString.stringWithUTF8String_(model_filename)
            options = InterpreterOptions.alloc().init()
            if num_threads is not None:
                options.numberOfThreads = num_threads

            self.interpreter = Interpreter.alloc().initWithModelPath_options_error_(
                model, options, self.error
            )
            self.allocate_tensors()

            if label_file:
                self.pose_class_names = self.load_labels(label_file)

            if label_file==None:
                self.input_height = self.input_shape[1]
                self.input_width = self.input_shape[2]

        def allocate_tensors(self):
            self.interpreter.allocateTensorsWithError_(self.error)

            self.input_shape = self.interpreter.inputTensorAtIndex_error_(
                0, self.error).shapeWithError_(self.error)
            self.input_shape = [
                self.input_shape.objectAtIndex_(_).intValue()
                for _ in range(self.input_shape.count())
            ]

            self.output_shape = self.interpreter.outputTensorAtIndex_error_(
                0, self.error).shapeWithError_(self.error)
            self.output_shape = [
                self.output_shape.objectAtIndex_(_).intValue()
                for _ in range(self.output_shape.count())
            ]

            self.output_type = self.interpreter.outputTensorAtIndex_error_(
                0, self.error).dataType

        def load_labels(self, label_path):
            with open(label_path, 'r') as f:
                return [line.strip() for _, line in enumerate(f.readlines())]

        def get_crop_size(self):
            return (self.input_height, self.input_width)

        def classify_pose(self, input_tensor):
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
            # assumes one input and one output for now
            bytestr = x.tobytes()

            # must cast to ctype._SimpleCData so that pyobjus passes pointer
            floatbuf = cast(bytestr, POINTER(c_float)).contents
            data = NSData.dataWithBytes_length_(floatbuf, len(bytestr))

            self.interpreter.copyData_toInputTensorAtIndex_error_(data, 0, self.error)
            self.interpreter.invokeWithError_(self.error)
            output = self.interpreter.outputTensorAtIndex_error_(0, self.error).dataWithError_(self.error).bytes()

            # have to do this to avoid memory leaks...
            while data.retainCount() > 1:
                data.release()

            return np.reshape(
                np.frombuffer(
                    (
                        c_float * np.prod(self.output_shape)).from_address(output.arg_ref), 
                        c_float
                    ), 
                    self.output_shape
                )

else:
    import tensorflow as tf

    class TensorFlowModel:
        def load(self, model_filename, num_threads=None, label_file=None):
            self.interpreter = tf.lite.Interpreter(model_filename, num_threads=num_threads)
            self.interpreter.allocate_tensors()

            self.input_index = self.interpreter.get_input_details()[0]['index']
            self.output_index = self.interpreter.get_output_details()[0]['index']

            if label_file==None:
                self.input_height = self.interpreter.get_input_details()[0]['shape'][1]
                self.input_width = self.interpreter.get_input_details()[0]['shape'][2]

            if label_file:
                self.pose_class_names = self.load_labels(label_file)

        def load_labels(self, label_path):
            with open(label_path, 'r') as f:
                return [line.strip() for _, line in enumerate(f.readlines())]

        def get_crop_size(self):
            return (self.input_height, self.input_width)

        def classify_pose(self, input_tensor):
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
            self.interpreter.set_tensor( self.input_index, x )
            self.interpreter.invoke()
            return self.interpreter.get_tensor( self.output_index )
