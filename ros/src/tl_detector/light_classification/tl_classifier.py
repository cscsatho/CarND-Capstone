from styx_msgs.msg import TrafficLight
import os, cv2, glob, time
import numpy as np
import tensorflow as tf
from collections import Counter
from utilities import label_map_util
from utilities import visualization_utils


def most_common(items):
    data = Counter(items)
    return max(items, key=data.get)


class TLClassifier(object):
    def __init__(self, draw_box=False):
        self.draw_box = draw_box
        self.score_thresh = 0.5
        self.skip_cnt = 0
        self.max_cnt = 4
        self.last_light = TrafficLight.UNKNOWN

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        model = curr_dir + '/model/frozen_inference_graph.pb'
        labels_file = curr_dir + '/model/label_map.pbtxt'

        label_map = label_map_util.load_labelmap(labels_file)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=4, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.graph = tf.Graph()

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.graph)
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        print("Loaded frozen model graph")


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        self.skip_cnt += 1
        if self.skip_cnt >= self.max_cnt:

            start = time.time()

            image_expanded = np.expand_dims(image, axis=0)
            with self.graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                    [self.boxes, self.scores, self.classes, self.num_detections],
                    feed_dict={self.image_tensor: image_expanded})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            if self.draw_box == True:
                print((scores, classes))

            lights = []
            for i in range(boxes.shape[0]):
                if scores[i] > self.score_thresh:
                    class_name = self.category_index[classes[i]]['name']

                    if class_name == 'Red':
                        traffic_light = TrafficLight.RED
                    elif class_name == 'Green':
                        traffic_light = TrafficLight.GREEN
                    elif class_name == 'Yellow':
                        traffic_light = TrafficLight.YELLOW

                    lights.append(traffic_light)

                    if self.draw_box == True:
                        boxlabel = '{}: {}%'.format(class_name, int(100*scores[i]))
                        visualization_utils.draw_bounding_box_on_image_array(image,
                            boxes[i][0], boxes[i][1],
                            boxes[i][2], boxes[i][3],
                            color=class_name,
                            thickness=4,
                            display_str_list=([boxlabel]),
                            use_normalized_coordinates=True)


            if self.draw_box == True:
                print('elapsed: ', time.time() - start)
                self.last_image = image

            lights.append(TrafficLight.UNKNOWN)
            self.last_light = most_common(lights)
            self.skip_cnt = 0

        return self.last_light


if __name__ == '__main__':

    test_imgs = glob.glob('test_data/*.png')

    light_classifier = TLClassifier(draw_box=True)
    for i, n in enumerate(test_imgs):
        img = cv2.imread(n)
        l = light_classifier.get_classification(img)
        cv2.imwrite(n[:-4]+'_.jpg', light_classifier.last_image)
        print(l)

