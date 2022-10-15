import os
import cv2
import numpy as np
import argparse

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import save_instances
from mrcnn.utils import download_trained_weights


class ProjectConfig(Config): # a subclass of Config, overriding the settings we need
     NAME = 'project'

     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 81 # the COCO dataset has 80 classes, and we also have to count the background

class Model:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.project_dir, 'logs')
        self.image_dir = os.path.join(self.project_dir, 'images')
        self.result_dir = os.path.join(self.project_dir, 'results')

        self.coco_model_path = os.path.join(self.project_dir, 'mask_rcnn_coco.h5')
        if not os.path.exists(self.coco_model_path):
            download_trained_weights(self.coco_model_path)

        self.model = MaskRCNN(mode = 'inference', model_dir = self.model_dir, config = ProjectConfig()) # create a model object, in inference mode
        self.model.load_weights(self.coco_model_path, by_name=True)

        # COCO class names
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                            'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear', 'hair drier', 'toothbrush']

    def open_image(self, file_name):
        file_path = os.path.join(self.image_dir, file_name)

        image_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return image

    def infer(self, image): # run detection
        results = self.model.detect([image], verbose = 1) # the model returns a list of dicts (one per image) with the bounding boxes, masks, plus classes and probabilities for them

        return results[0] # since we're only detecting one image at a time, the dict has a single element, which we return


    def save_visualization(self, file_name, image, r): # save a visualization of the image with the masks and bounding boxes overlaid on top
        file_path = os.path.join(self.result_dir, file_name + '_detected.jpg')

        save_instances(file_path, image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])

    def save_segments(self, file_name, image, masks, use_grabcut): # use the masks from the detection to create the different segments, and save them
        for i in range(masks.shape[2]): # masks is a numpy array: [height, width, mask_index], so we loop over their indices to apply each mask separately
            temp = image.copy()

            if use_grabcut:
                grabcut_mask = self.apply_grabcut(image, masks[:,:,i]) # create a new grabcut mask using the i-th RCNN mask as a starting point
                temp = cv2.bitwise_and(temp, temp, mask = grabcut_mask) # apply the grabcut mask

                file_path = os.path.join(self.result_dir, file_name + '_grabcut_segment' + str(i) + '.jpg')
            else:
                temp = cv2.bitwise_and(temp, temp, mask = masks[:,:,i]) # apply the i-th mask

                file_path = os.path.join(self.result_dir, file_name + '_segment' + str(i) + '.jpg')

            cv2.imwrite(file_path, cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))       

    def apply_grabcut(self, image, mask):
        temp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # grabCut works with BGR images

        # two zero-filled arrays that GrabCut internally uses have to be passed as args
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)

        grabcut_mask = np.copy(mask)
        grabcut_mask[grabcut_mask > 0] = cv2.GC_PR_FGD # probable foreground
        grabcut_mask[grabcut_mask == 0] = cv2.GC_BGD # background

        grabcut_mask, bgModel, fgModel = cv2.grabCut(temp, grabcut_mask, None, bgModel, fgModel, iterCount = 5, mode = cv2.GC_INIT_WITH_MASK)

        # the new mask has 4 possible values for each pixel, but we only need the foreground, so we
        # we set all definite and probable background pixels to 0 and definite and probable foreground pixels to 1
        return np.where((grabcut_mask == cv2.GC_BGD) | (grabcut_mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)


    def process_image(self, file_name, use_grabcut):
        image = self.open_image(file_name)

        print(file_name)      
        r = self.infer(image)

        self.save_visualization(file_name, image, r)
        self.save_segments(file_name, image, r['masks'].astype(np.uint8), use_grabcut) # the masks are boolean, so we convert them to ints       

    def process_folder_sequence(self, start, end):
        file_names = next(os.walk(self.image_dir))[2]

        for file_name in file_names[start:end]:
            self.process_image(file_name, False)

    def process_folder(self):
        file_names = next(os.walk(self.image_dir))[2]

        for file_name in file_names:
            self.process_image(file_name, False)


    def check_for_image(self, file_name):
        path = os.path.join(self.image_dir, file_name) 

        return os.path.isfile(path)


ap = argparse.ArgumentParser()
ap.add_argument('--use_grabcut', action = 'store_true', help = 'try to refine the segmentation by additionally applying grabCut to the mask produced by MaskRCNN')
args = vars(ap.parse_args())

model = Model()

print('Enter a filename (of an image from the images folder), or EXIT to stop.')
while True:
    data = input('image: ')

    if model.check_for_image(data):
        model.process_image(data, args['use_grabcut'])    
    elif data == 'EXIT':
        break
    else:
        continue


#ap.add_argument('--image', required = True, help = 'path to image to be segmented')
#model.process_image(args['image'], args['use_grabcut'])