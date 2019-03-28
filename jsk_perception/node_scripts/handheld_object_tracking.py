#!/usr/bin/env python

import cv2
import numpy as np
import chainer
from chainer.links import caffe
import chainer.functions as F

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from geometry_msgs.msg import PolygonStamped as Rect

(CV_MAJOR, CV_MINOR, _) = cv2.__version__.split(".")


class HandHheldObjectTracking(object):

    def __init__(self):
        self.cv_bridge = CvBridge()

        pretrained_model = rospy.get_param('~pretrained_model')
        self.model = caffe.caffe_function.CaffeFunction(
            pretrained_model)

        self.gpu = rospy.get_param('~gpu', 0)
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

        self.use_area = rospy.get_param('~use_area', True)
        self.max_distance = rospy.get_param('~max_distance', 1.5)

        self.scales = np.array([1.250], dtype=np.float32)

        self.rect = None
        self.batch_size = int(self.scales.shape[0])
        self.n_channel = 6
        self.image_width = 227
        self.image_height = 227

        self.prev_roi = None
        self.prev_rgb = None
        self.prev_depth = None

        self.mask_pub = rospy.Publisher(
            '~output/mask', Image, queue_size=1)
        self.image_pub = rospy.Publisher('~output/viz', Image, queue_size=1)
        self.debug_image_pub = rospy.Publisher('~output/debug/viz',
                                               Image, queue_size=1)
        self.rect_pub = rospy.Publisher('~output/rect', Rect, queue_size=1)

        self.subscribe()

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 100)
        self.sub_rect = rospy.Subscriber(
            '~input/rect', Rect,
            self.screen_point_callback)
        sub_img = message_filters.Subscriber(
            '~input', Image, queue_size=1)
        sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1)
        self.subs = [sub_img, sub_depth]
        if rospy.get_param('~approximate_sync', True):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self.callback)

    def normalize_data(self, rgb, depth):
        # normalize and encode
        rgb = rgb.astype(np.float32)
        rgb /= rgb.max()
        rgb = (rgb - rgb.min())/(rgb.max() - rgb.min())

        depth = depth.astype(np.float32) \
            if depth.dtype is not str('float32') else depth

        depth /= depth.max()
        depth *= 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

        depth = depth.astype(np.float32)
        depth /= depth.max()
        depth = (depth - depth.min())/(depth.max() - depth.min())

        return rgb, depth

    def process_rgbd(self, rgb, depth, rect, scale=1.5):
        # crop (build multiple scale)
        rect = self.get_region_bbox(rgb, rect, scale)
        x, y, w, h = rect
        rgb = rgb[y:y+h, x:x+w].copy()
        depth = depth[y:y+h, x:x+w].copy()

        image = rgb.copy()

        # resize to network input
        rgb = cv2.resize(
            rgb, (self.image_width, self.image_height))
        depth = cv2.resize(
            depth, (self.image_width, self.image_height))

        # transpose to c, h, w
        rgb = rgb.transpose((2, 0, 1))
        depth = depth.transpose((2, 0, 1))

        return rgb, depth, image, rect

    def track(self, rgb, depth, header=None):
        dist_mask_thresh = 4
        dist_mask_thresh *= 1000.0 if depth.max() > 1000.00 else 1.0

        depth[depth > dist_mask_thresh] = 0.0  # mask depth
        im_nrgb, im_ndep = self.normalize_data(
            rgb, depth)  # normalize data

        if self.prev_depth is None or \
           self.prev_rgb is None or \
           self.prev_roi is None:
            self.prev_rgb = im_nrgb.copy()
            self.prev_depth = im_ndep.copy()
            self.prev_roi = self.rect

        crop_rects = []
        target_data = np.zeros(
            (self.batch_size, self.n_channel,
             self.image_height, self.image_width), 'f')
        template_data = np.zeros_like(target_data, 'f')
        for index, scale in enumerate(self.scales):
            in_rgb, in_dep, image, rect = self.process_rgbd(
                im_nrgb, im_ndep,
                self.rect.copy(), scale)
            target_data[index][0:3, :, :] = in_rgb.copy()
            target_data[index][3:6, :, :] = in_dep.copy()
            crop_rects.append(rect)

            # template cropping
            in_rgb, in_dep, image, prect = self.process_rgbd(
                self.prev_rgb, self.prev_depth,
                self.prev_roi.copy(), scale)
            target_data[index][0:3, :, :] = in_rgb.copy()
            target_data[index][3:6, :, :] = in_dep.copy()

        if self.gpu >= 0:
            target_data = chainer.cuda.to_gpu(target_data)
            template_data = chainer.cuda.to_gpu(template_data)
        output = F.softmax(self.model(inputs={'target_data': target_data,
                                              'template_data': template_data},
                                      outputs=['upscore'])[0]).data
        if self.gpu >= 0:
            output = chainer.cuda.to_cpu(output)

        update_model = True
        tmp_rect = self.rect.copy()

        probability_map = []

        for index in xrange(0, self.batch_size, 1):
            feat = output[index]
            prob = feat[1].copy()
            prob *= 255
            prob = prob.astype(np.uint8)

            rect = crop_rects[index]
            prob = cv2.resize(prob, (rect[2], rect[3]))
            probability_map.append(prob)

        for index, prob in enumerate(probability_map):
            kernel = np.ones((7, 7), np.uint8)
            prob = cv2.erode(prob, kernel, iterations=1)

            prob = cv2.GaussianBlur(prob, (5, 5), 0)
            _, prob = cv2.threshold(
                prob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            rect = crop_rects[index]
            bbox = self.create_mask_rect(prob, rect)
            if bbox is None:
                update_model = False
                return

            bbox = np.array(bbox, dtype=np.int)
            bbox[0] += rect[0]
            bbox[1] += rect[1]

            # enlarge by padding
            bbox = self.get_bbox(rgb, bbox, 10)

            im_mask = np.zeros(
                (rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            x, y, w, h = crop_rects[index]
            im_mask[y:y+h, x:x+w] = prob

            self.rect = bbox.copy()

            x, y, w, h = bbox
            cv2.rectangle(rgb, (int(x), int(y)),
                          (int(x+w), int(h+y)), (0, 255, 0), 4)

            # test
            kernel = np.ones((9, 9), np.uint8)
            im_mask = cv2.dilate(im_mask, kernel, iterations=1)

            debug_image = cv2.addWeighted(
                rgb, 0.5, cv2.cvtColor(im_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
            debug_image_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_image_msg.header = header
            self.debug_image_pub.publish(debug_image_msg)

            # remove incorrect mask by depth masking
            im_mask = self.depth_mask_filter(depth, im_mask, self.max_distance)

            # reduce mask by scale
            mask_msg = self.cv_bridge.cv2_to_imgmsg(im_mask, "mono8")
            mask_msg.header = header
            self.mask_pub.publish(mask_msg)
            image_msg = self.cv_bridge.cv2_to_imgmsg(rgb, "bgr8")
            image_msg.header = header
            self.image_pub.publish(image_msg)

            rect_msg = Rect()
            rect_msg.header = header
            pt = Point32()
            pt.x = bbox[0]
            pt.y = bbox[1]
            rect_msg.polygon.points.append(pt)
            pt = Point32()
            pt.x = bbox[2] + bbox[0]
            pt.y = bbox[3] + bbox[1]
            rect_msg.polygon.points.append(pt)
            self.rect_pub.publish(rect_msg)

        if update_model:
            self.prev_rgb = im_nrgb.copy()
            self.prev_depth = im_ndep.copy()
            self.prev_roi = tmp_rect

    def depth_mask_filter(self, depth, im_mask, max_dist=1.5):
        im_mask[depth > max_dist] = 0
        return im_mask

    def create_mask_rect(self, im_gray, rect):  # ! rect used for cropping
        if len(im_gray.shape) is None:
            return

        if CV_MAJOR < '3':
            contour, _ = cv2.findContours(
                im_gray.copy(),
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
        else:
            im, contour, _ = cv2.findContours(
                im_gray.copy(), cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)

        prev_center = np.array([self.rect[0] + self.rect[2] / 2.0,
                                self.rect[1] + self.rect[3] / 2.0])

        max_area = 0
        min_distance = float('inf')
        index = -1
        for i, cnt in enumerate(contour):
            if not self.use_area:
                box = cv2.boundingRect(contour[index])
                center = np.array([box[0] + rect[0] + box[2] / 2.0,
                                   box[1] + rect[1] + box[3] / 2.0])
                distance = np.linalg.norm(prev_center - center)

                if distance < min_distance:
                    min_distance = distance
                    index = i

            a = cv2.contourArea(cnt)
            if max_area < a:
                max_area = a
                index = i

        rect = cv2.boundingRect(contour[index]) if index > -1 else None
        return rect

    def get_bbox(self, rgb, rect, pad=8):
        x, y, w, h = rect

        nx = int(x - pad)
        ny = int(y - pad)
        nw = int(w + (2 * pad))
        nh = int(h + (2 * pad))

        nx = 0 if nx < 0 else nx
        ny = 0 if ny < 0 else ny
        nw = nw-((nx+nw)-rgb.shape[1]) if (nx+nw) > rgb.shape[1] else nw
        nh = nh-((ny+nh)-rgb.shape[0]) if (ny+nh) > rgb.shape[0] else nh

        return np.array([nx, ny, nw, nh])

    def callback(self, image_msg, depth_msg):
        rgb = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        if depth_msg.encoding == '16UC1':
            depth = np.asarray(depth, dtype=np.float32)
            depth /= 1000  # convert metric: mm -> m
        elif depth_msg.encoding != '32FC1':
            rospy.logerr('Unsupported depth encoding: %s' % depth_msg.encoding)

        if rgb is None or depth is None:
            rospy.logwarn('input msg is empty')
            return

        depth[np.isnan(depth)] = 0.0

        if self.rect is not None:
            self.track(rgb, depth, image_msg.header)
        else:
            rospy.loginfo_throttle(60, 'Object not initialized ...')

    def get_region_bbox(self, rgb, rect, scale=1.5):
        x, y, w, h = rect
        cx, cy = (x + w/2.0, y + h/2.0)
        s = scale

        nw = int(s * w)
        nh = int(s * h)
        nx = int(cx - nw/2.0)
        ny = int(cy - nh/2.0)

        nx = 0 if nx < 0 else nx
        ny = 0 if ny < 0 else ny
        nw = nw-((nx+nw)-rgb.shape[1]) if (nx+nw) > rgb.shape[1] else nw
        nh = nh-((ny+nh)-rgb.shape[0]) if (ny+nh) > rgb.shape[0] else nh

        return np.array([nx, ny, nw, nh])

    def screen_point_callback(self, rect_msg):
        x = rect_msg.polygon.points[0].x
        y = rect_msg.polygon.points[0].y
        w = rect_msg.polygon.points[1].x - x
        h = rect_msg.polygon.points[1].y - y
        self.prev_roi = None
        self.rect = np.array([x, y, w, h])
        rospy.loginfo('Object Rect Received')


def main():
    rospy.init_node('handheld_object_tracking')
    hhot = HandHheldObjectTracking()  # NOQA
    rospy.spin()


if __name__ == '__main__':
    main()
