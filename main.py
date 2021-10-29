from detection import *

detector = Detector(model_type = "OD")

path = "test_data/outdoor.mp4"
# detector.onImage(path)

detector.on_Video(path)

