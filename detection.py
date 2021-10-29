from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import time

class Detector:
	def __init__(self,model_type):
		self.model_type = model_type

		self.cfg = get_cfg()
		if self.model_type == "OD":
			self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
		elif self.model_type == "IS":
			self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		elif self.model_type == "PS":
			self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")

		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
		self.cfg.MODEL.DEVICE = "cpu"

		self.predictor = DefaultPredictor(self.cfg)

	def onImage(self,imagepath):
		image = cv2.imread(imagepath)
		image = cv2.resize(image,(640,420), interpolation = cv2.INTER_AREA)
		if self.model_type != "PS":
			predictions = self.predictor(image)


			viz = Visualizer(image[:,:,::-1] , metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN
				[0]), instance_mode = ColorMode.SEGMENTATION)

			output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
		else:
			predictions , segmentinfo = self.predictor(image)["panoptic_seg"]
		
			viz = Visualizer(image[:,:,::-1] , metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN
				[0]))
			output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"),segmentinfo)

		cv2.imshow("Result", output.get_image()[:,:,::-1])
		cv2.imwrite("image/fp4_ps.jpg",output.get_image()[:,:,::-1])
		cv2.waitKey(0)

	def on_Video(self,path):
		cap = cv2.VideoCapture(path)

		fps = cap.get(cv2.CAP_PROP_FPS)
		print("Frames per second camera: {0}".format(fps))

		# Number of frames to capture
		num_frames = 1;

		print("Capturing {0} frames".format(num_frames))

		if(cap.isOpened() == False):
			print("Error opening file.")
			return

		(success,image) = cap.read()

		while success:

			# Start time
			start = time.time()

			image = cv2.resize(image,(640,420), interpolation = cv2.INTER_AREA)
			if self.model_type != "PS":
				predictions = self.predictor(image)

				viz = Visualizer(image[:,:,::-1] , metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN
					[0]), instance_mode = ColorMode.SEGMENTATION)

				output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
			else:
				predictions , segmentinfo = self.predictor(image)["panoptic_seg"]
			
				viz = Visualizer(image[:,:,::-1] , metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN
					[0]))
				output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"),segmentinfo)

			end = time.time()

			# Time elapsed
			seconds = end - start
			#print ("Time taken : {0} seconds".format(seconds))

			# Calculate frames per second
			fps  = num_frames / seconds
			print(fps)

			cv2.putText(output.get_image()[:,:,::-1], "FPS: " + str(round(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255))

			cv2.imshow("Result",output.get_image()[:,:,::-1])

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		capture.release()
		cv2.destroyAllWindows()





	
