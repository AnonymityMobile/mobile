class Config:
    def __init__(self):

        self.end_model_format = ['.tflite', '.lite', '.pt', '.ptl', '.param', '.mlmodel', '.model', '.caffemodel',
                                 '.feathermodel', '.chainermodel', 'PaddlePredictor.jar', 'libpaddle_lite_jni.so',
                                 '.nnet', 'libtvm_rumtime.so', '.moa', 'model.prof',
                                 '.mallet', '.classifier', '.inferencer', '.cntk']

        self.path_result = 'result.txt'
