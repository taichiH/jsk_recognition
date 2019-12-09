#!/usr/bin/env python
# -*- coding:utf-8 -*-

#    See: https://arxiv.org/abs/1611.08050

import math

import chainer
import chainer.functions as F
from chainer import cuda
import cv2
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pylab as plt  # NOQA

import cv_bridge
import message_filters
import rospy
from jsk_topic_tools import ConnectionBasedTransport
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from jsk_recognition_msgs.msg import HumanSkeleton
from jsk_recognition_msgs.msg import HumanSkeletonArray
from jsk_recognition_msgs.msg import PeoplePose
from jsk_recognition_msgs.msg import PeoplePoseArray
from jsk_recognition_msgs.msg import Segment
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from openpose import PoseNet, HandNet


def find_joint(limb, jts):
    jt = [jt for jt in jts if jt['limb'] == limb]
    if jt:
        return jt[0]
    else:
        return None


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


class PeoplePoseEstimation2D(ConnectionBasedTransport):
    # Note: the order of this sequences is important
    # find connection in the specified sequence,
    # center 29 is in the position 15
    limb_sequence = [[ 2,  1], [ 1, 16], [ 1, 15], [ 6, 18], [ 3, 17],
                     [ 2,  3], [ 2,  6], [ 3,  4], [ 4,  5], [ 6,  7],
                     [ 7,  8], [ 2,  9], [ 9, 10], [10, 11], [ 2, 12],
                     [12, 13], [13, 14], [15, 17], [16, 18]]
    # the middle joints heatmap correpondence
    map_idx = [[47, 48], [49, 50], [51, 52], [37, 38], [45, 46],
               [31, 32], [39, 40], [33, 34], [35, 36], [41, 42],
               [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
               [27, 28], [29, 30], [53, 54], [55, 56]]
    # length ratio from connections
    limb_length_hand_ratio = [ 0.6,  0.2,  0.2, 0.85, 0.85,
                               0.6,  0.6, 0.93, 0.65, 0.95,
                              0.65,  2.2,  1.7,  1.7,  2.2,
                               1.7,  1.7, 0.25, 0.25]
    # hand joint connection sequence
    hand_sequence = [[0, 1],   [1, 2],   [2, 3],   [3, 4],
                     [0, 5],   [5, 6],   [6, 7],   [7, 8],
                     [0, 9],   [9, 10],  [10, 11], [11, 12],
                     [0, 13],  [13, 14], [14, 15], [15, 16],
                     [0, 17],  [17, 18], [18, 19], [19, 20],]

    index2limbname = ["Nose",
                      "Neck",
                      "RShoulder",
                      "RElbow",
                      "RWrist",
                      "LShoulder",
                      "LElbow",
                      "LWrist",
                      "RHip",
                      "RKnee",
                      "RAnkle",
                      "LHip",
                      "LKnee",
                      "LAnkle",
                      "REye",
                      "LEye",
                      "REar",
                      "LEar",
                      "Bkg"]

    index2handname = ["RHand{}".format(i) for i in range(21)] +\
                     ["LHand{}".format(i) for i in range(21)]

    def __init__(self):
        super(self.__class__, self).__init__()
        self.backend = rospy.get_param('~backend', 'chainer')
        self.scales = rospy.get_param('~scales', [0.38])
        self.stride = rospy.get_param('~stride', 8)
        self.pad_value = rospy.get_param('~pad_value', 128)
        self.thre1 = rospy.get_param('~thre1', 0.1)
        self.thre2 = rospy.get_param('~thre2', 0.05)
        self.width = rospy.get_param('~width', None)
        self.height = rospy.get_param('~height', None)
        self.check_wh()
        self.gpu = rospy.get_param('~gpu', -1)  # -1 is cpu mode
        self.with_depth = rospy.get_param('~with_depth', False)
        # hand detection
        self.use_hand = rospy.get_param('~hand/enable', False)
        self.hand_gaussian_ksize = rospy.get_param('~hand/gaussian_ksize', 17)
        self.hand_gaussian_sigma = rospy.get_param('~hand/gaussian_sigma', 2.5)
        self.hand_thre1 = rospy.get_param('~hand/thre1', 20)
        self.hand_thre2 = rospy.get_param('~hand/thre2', 0.1)
        self.hand_width_offset = rospy.get_param('~hand/width_offset', 0)
        # model loading
        self._load_model()
        # topic advertise
        self.image_pub = self.advertise('~output', Image, queue_size=1)
        self.pose_pub = self.advertise('~pose', PeoplePoseArray, queue_size=1)
        self.sub_info = None
        if self.with_depth is True:
            self.pose_2d_pub = self.advertise('~pose_2d', PeoplePoseArray, queue_size=1)
            # visualization rviz plugin: https://github.com/jsk-ros-pkg/jsk_visualization/pull/740
            self.skeleton_pub = self.advertise(
                '~skeleton', HumanSkeletonArray, queue_size=1)

    def check_wh(self):
        if (self.width is None) != (self.height is None):
            rospy.logwarn('width and height should be specified, but '
                          'specified only {}'
                          .format('height' if self.height else 'width'))

    @property
    def visualize(self):
        return self.image_pub.get_num_connections() > 0

    def _load_model(self):
        if self.backend == 'chainer':
            self._load_chainer_model()
        else:
            raise RuntimeError('Unsupported backend: %s', self.backend)

    def _load_chainer_model(self):
        # model_file = rospy.get_param('~model_file')
        # self.pose_net = PoseNet(pretrained_model=model_file)
        # rospy.loginfo('Finished loading trained model: {0}'.format(model_file))
        # hand net
        # if self.use_hand:
        model_file = rospy.get_param('~hand/model_file')
        self.hand_net = HandNet(pretrained_model=model_file)
        rospy.loginfo('Finished loading trained hand model: {}'.format(model_file))
        #
        if self.gpu >= 0:
            # self.pose_net.to_gpu(self.gpu)
            # if self.use_hand:
            self.hand_net.to_gpu(self.gpu)
            # create gaussian kernel
            ksize = self.hand_gaussian_ksize
            sigma = self.hand_gaussian_sigma
            c = ksize // 2
            k = np.zeros((1, 1, ksize, ksize), dtype=np.float32)
            for y in range(ksize):
                dy = abs(y - c)
                for x in range(ksize):
                    dx = abs(x - c)
                    e = np.exp(- (dx ** 2 + dy ** 2) / (2 * sigma ** 2))
                    k[0][0][y][x] = 1 / (sigma ** 2 * 2 * np.pi) * e
            k = chainer.cuda.to_gpu(k, device=self.gpu)
            self.hand_gaussian_kernel = k
        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

    def subscribe(self):
        if self.with_depth:
            queue_size = rospy.get_param('~queue_size', 10)
            sub_img = message_filters.Subscriber(
                '~input', Image, queue_size=1, buff_size=2**24)
            sub_depth = message_filters.Subscriber(
                '~input/depth', Image, queue_size=1, buff_size=2**24)
            self.subs = [sub_img, sub_depth]

            # NOTE: Camera info is not synchronized by default.
            # See https://github.com/jsk-ros-pkg/jsk_recognition/issues/2165
            sync_cam_info = rospy.get_param("~sync_camera_info", False)
            if sync_cam_info:
                sub_info = message_filters.Subscriber(
                    '~input/info', CameraInfo, queue_size=1, buff_size=2**24)
                self.subs.append(sub_info)
            else:
                self.sub_info = rospy.Subscriber(
                    '~input/info', CameraInfo, self._cb_cam_info)

            if rospy.get_param('~approximate_sync', True):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            if sync_cam_info:
                sync.registerCallback(self._cb_with_depth_info)
            else:
                self.camera_info_msg = None
                sync.registerCallback(self._cb_with_depth)
        else:
            queue_size = rospy.get_param('~queue_size', 10)
            sub_img = message_filters.Subscriber(
                '~input', Image, queue_size=1, buff_size=2**24)
            sub_rect = message_filters.Subscriber(
                '~input/rect', RectArray, queue_size=1, buff_size=2*24)
            self.subs = [sub_img, sub_rect]

            if rospy.get_param('~approximate_sync', True):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            sync.registerCallback(self._cb_with_rect)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
        if self.sub_info is not None:
            self.sub_info.unregister()
            self.sub_info = None

    def _cb_cam_info(self, msg):
        self.camera_info_msg = msg
        self.sub_info.unregister()
        self.sub_info = None
        rospy.loginfo("Received camera info")

    def _cb_with_depth(self, img_msg, depth_msg):
        if self.camera_info_msg is None:
            return
        self._cb_with_depth_info(img_msg, depth_msg, self.camera_info_msg)

    def _cb_with_depth_info(self, img_msg, depth_msg, camera_info_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth_img = br.imgmsg_to_cv2(depth_msg, 'passthrough')
        if depth_msg.encoding == '16UC1':
            depth_img = np.asarray(depth_img, dtype=np.float32)
            depth_img /= 1000  # convert metric: mm -> m
        elif depth_msg.encoding != '32FC1':
            rospy.logerr('Unsupported depth encoding: %s' % depth_msg.encoding)

        people_joint_positions, all_peaks = self.pose_estimate(img)
        if self.use_hand:
            people_joint_positions = self.hand_estimate(
                img, people_joint_positions)

        people_pose_msg = PeoplePoseArray()
        people_pose_msg.header = img_msg.header
        people_pose_2d_msg = self._create_2d_people_pose_array_msgs(
            people_joint_positions,
            img_msg.header)
        skeleton_msgs = HumanSkeletonArray(header=img_msg.header)

        # calculate xyz-position
        fx = camera_info_msg.K[0]
        fy = camera_info_msg.K[4]
        cx = camera_info_msg.K[2]
        cy = camera_info_msg.K[5]
        for person_joint_positions in people_joint_positions:
            pose_msg = PeoplePose()
            skeleton_msg = HumanSkeleton(header=img_msg.header)
            for joint_pos in person_joint_positions:
                if joint_pos['score'] < 0:
                    continue
                if 0 <= joint_pos['y'] < depth_img.shape[0] and\
                   0 <= joint_pos['x'] < depth_img.shape[1]:
                    z = float(depth_img[int(joint_pos['y'])][int(joint_pos['x'])])
                else:
                    continue
                if np.isnan(z):
                    continue
                x = (joint_pos['x'] - cx) * z / fx
                y = (joint_pos['y'] - cy) * z / fy
                pose_msg.limb_names.append(joint_pos['limb'])
                pose_msg.scores.append(joint_pos['score'])
                pose_msg.poses.append(Pose(position=Point(x=x, y=y, z=z),
                                           orientation=Quaternion(w=1)))
            people_pose_msg.poses.append(pose_msg)

            for i, conn in enumerate(self.limb_sequence):
                j1_name = self.index2limbname[conn[0] - 1]
                j2_name = self.index2limbname[conn[1] - 1]
                if j1_name not in pose_msg.limb_names \
                        or j2_name not in pose_msg.limb_names:
                    continue
                j1_index = pose_msg.limb_names.index(j1_name)
                j2_index = pose_msg.limb_names.index(j2_name)
                bone_name = '{}->{}'.format(j1_name, j2_name)
                bone = Segment(
                    start_point=pose_msg.poses[j1_index].position,
                    end_point=pose_msg.poses[j2_index].position)
                skeleton_msg.bones.append(bone)
                skeleton_msg.bone_names.append(bone_name)
            skeleton_msgs.skeletons.append(skeleton_msg)

        self.pose_2d_pub.publish(people_pose_2d_msg)
        self.pose_pub.publish(people_pose_msg)
        self.skeleton_pub.publish(skeleton_msgs)

        if self.visualize:
            vis_img = self._draw_joints(img, people_joint_positions, all_peaks)
            vis_msg = br.cv2_to_imgmsg(vis_img, encoding='bgr8')
            vis_msg.header.stamp = img_msg.header.stamp
            self.image_pub.publish(vis_msg)

    def _cb(self, img_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        people_joint_positions, all_peaks = self.pose_estimate(img)
        if self.use_hand:
            people_joint_positions = self.hand_estimate(
                img, people_joint_positions)

        people_pose_msg = self._create_2d_people_pose_array_msgs(
            people_joint_positions,
            img_msg.header)

        self.pose_pub.publish(people_pose_msg)

        if self.visualize:
            vis_img = self._draw_joints(img, people_joint_positions, all_peaks)
            vis_msg = br.cv2_to_imgmsg(vis_img, encoding='bgr8')
            vis_msg.header.stamp = img_msg.header.stamp
            self.image_pub.publish(vis_msg)

    def _cb_with_rect(self, img_msg, rect_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        rects = rect_msg.rects

        people_joint_positions = self.hand_estimate_with_rects(
            img, rects)

        people_pose_msg = self._create_2d_people_pose_array_msgs(
            people_joint_positions,
            img_msg.header)

        self.pose_pub.publish(people_pose_msg)

        if self.visualize:
            vis_img = self._draw_hand_joints(img, people_joint_positions)
            vis_msg = br.cv2_to_imgmsg(vis_img, encoding='bgr8')
            vis_msg.header.stamp = img_msg.header.stamp
            self.image_pub.publish(vis_msg)

    def _create_2d_people_pose_array_msgs(self, people_joint_positions, header):
        people_pose_msg = PeoplePoseArray(header=header)
        for person_joint_positions in people_joint_positions:
            pose_msg = PeoplePose()
            for joint_pos in person_joint_positions:
                if joint_pos['score'] < 0:
                    continue
                pose_msg.limb_names.append(joint_pos['limb'])
                pose_msg.scores.append(joint_pos['score'])
                pose_msg.poses.append(Pose(position=Point(x=joint_pos['x'],
                                                          y=joint_pos['y'],
                                                          z=0)))
            people_pose_msg.poses.append(pose_msg)
        return people_pose_msg

    def _draw_hand_joints(self, img, people_joint_positions):
        cmap = matplotlib.cm.get_cmap('hsv')
        stickwidth = 4
        offset = len(self.limb_sequence)
        limb_joint_positions_padding = []
        for limb in self.index2limbname:
            limb_joint_positions_padding.append({'x': 0, 'y': 0, 'score': -1, 'limb': limb})
        people_joint_positions = [limb_joint_positions_padding + joint_positions for joint_positions in people_joint_positions]

        for joint_positions in people_joint_positions:
            n = len(joint_positions[offset:])
            for i, jt in enumerate(joint_positions[offset:]):
                if jt['score'] < 0.0:
                    continue
                rgba = np.array(cmap(1. * i / n))
                color = rgba[:3] * 255
                cv2.circle(img, (int(jt['x']), int(jt['y'])),
                           2, color, thickness=-1)

        for joint_positions in people_joint_positions:
            offset = len(self.limb_sequence)
            n = len(self.hand_sequence)
            for _ in range(2):
                # for both hands
                for i, conn in enumerate(self.hand_sequence):
                    rgba = np.array(cmap(1. * i / n))
                    color = rgba[:3] * 255
                    j1 = joint_positions[offset + conn[0]]
                    j2 = joint_positions[offset + conn[1]]
                    if j1['score'] < 0 or j2['score'] < 0:
                        continue
                    cx, cy = int((j1['x'] + j2['x']) / 2.), int((j1['y'] + j2['y']) / 2.)
                    dx, dy = j1['x'] - j2['x'], j1['y'] - j2['y']
                    length = np.linalg.norm([dx, dy])
                    angle = int(np.degrees(np.arctan2(dy, dx)))
                    polygon = cv2.ellipse2Poly((cx, cy), (int(length / 2.), stickwidth),
                                               angle, 0, 360, 1)
                    top, left = np.min(polygon[:,1]), np.min(polygon[:,0])
                    bottom, right = np.max(polygon[:,1]), np.max(polygon[:,0])
                    roi = img[top:bottom,left:right]
                    roi2 = roi.copy()
                    cv2.fillConvexPoly(roi2, polygon - np.array([left, top]), color)
                    cv2.addWeighted(roi, 0.4, roi2, 0.6, 0.0, dst=roi)
                #
                offset += len(self.index2handname) / 2

        return img

    def hand_estimate_with_rects(self, bgr, rects):
        if self.backend == 'chainer':
            return self._hand_estimate_with_rects_chainer_backend(bgr, rects)
        raise ValueError('Unsupported backend: {0}'.format(self.backend))

    def _hand_estimate_with_rects_chainer_backend(self, bgr, rects):
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/29ea7e24dce4abae30faecf769855823ad7bb637/src/openpose/hand/handDetector.cpp
        people_joint_positions = []
        for i in range(len(rects)):
            people_joint_positions.append([])
        for joint_positions, rect in zip(people_joint_positions, rects):
            # crop hand image for each rect
            hand_joint_positions = []
            width = rect.width
            height = rect.height
            cx = rect.x + int(width/2.)
            cy = rect.y + int(height/2.)
            width = max(width, height) * 2.0
            if min(rect.width, rect.height) > self.hand_thre1:
                # hand_bgr = self._crop_square_image_with_rect(bgr, rect) # TODO: width != height
                hand_bgr = self._crop_square_image(bgr, cx, cy, width)
                hand_joints = self._hand_estimate_chainer_backend_each(
                    hand_bgr, cx, cy, False)
                hand_joint_positions.extend(hand_joints)
                if self.visualize:
                    cv2.circle(bgr, (int(cx), int(cy)), int(width/2.),
                               (255, 0, 0), thickness=1)
                    # cv2.ellipse(bgr, ((int(cx), int(cy)), (int(width*2.0), int(height*2.0)), 0), (255, 0, 0), thickness=1) # TODO: width != height

            for limb in self.index2handname:
                jt = find_joint(limb, hand_joint_positions)
                if jt is not None:
                    joint_positions.append(jt)
                else:
                    joint_positions.append({
                        'x': 0, 'y': 0, 'score': -1, 'limb': limb})

        return people_joint_positions

    def _hand_estimate_chainer_backend_each(self, hand_bgr, cx, cy, left_hand):
        xp = self.hand_net.xp

        if left_hand:
            hand_bgr = cv2.flip(hand_bgr, 1)  # 1 = vertical

        resized = cv2.resize(hand_bgr, (368, 368), interpolation=cv2.INTER_CUBIC)
        x = np.array(resized[np.newaxis], dtype=np.float32)
        x = x.transpose(0, 3, 1, 2)
        x = x / 256 - 0.5

        if self.gpu >= 0:
            x = chainer.cuda.to_gpu(x)
        x = chainer.Variable(x)

        heatmaps = self.hand_net(x)
        heatmaps = F.resize_images(heatmaps[-1], hand_bgr.shape[:2])[0]
        if self.gpu >= 0:
            heatmaps.to_cpu()
        heatmaps = heatmaps.array

        if left_hand:
            heatmaps = heatmaps.transpose(1, 2, 0)
            heatmaps = cv2.flip(heatmaps, 1)
            heatmaps = heatmaps.transpose(2, 0, 1)

        # get peak on heatmap
        hmaps = []
        if xp == np:
            # cpu
            for i in range(heatmaps.shape[0] - 1):
                heatmap = gaussian_filter(heatmaps[i], sigma=self.hand_gaussian_sigma)
                hmaps.append(heatmap)
        else:
            heatmaps = chainer.cuda.to_gpu(heatmaps)
            heatmaps = F.convolution_2d(
                heatmaps[:, xp.newaxis], self.hand_gaussian_kernel,
                stride=1, pad=int(self.hand_gaussian_ksize / 2))
            heatmaps = chainer.cuda.to_cpu(xp.squeeze(heatmaps.array))
            for heatmap in heatmaps[:-1]:
                hmaps.append(heatmap)
        keypoints = []
        idx_offset = 0
        if left_hand:
            idx_offset += len(hmaps)
        for i, heatmap in enumerate(hmaps):
            conf = heatmap.max()
            cds = np.array(np.where(heatmap==conf)).flatten().tolist()
            py = cy + cds[0] - hand_bgr.shape[0] / 2
            px = cx + cds[1] - hand_bgr.shape[1] / 2
            keypoints.append({'x': px, 'y': py, 'score': conf,
                              'limb': self.index2handname[idx_offset+i]})
        return keypoints

    def _crop_square_image(self, img, cx, cy, width):
        cx, cy, width = int(cx), int(cy), int(width)
        left, right = cx - int(width / 2), cx + int(width / 2)
        top, bottom = cy - int(width / 2), cy + int(width / 2)
        imh, imw, imc = img.shape
        cropped = img[max(0, top):max(min(imh, bottom), 0), max(0, left):max(min(imw, right), 0)]
        ch, cw = cropped.shape[:2]
        bx, by = max(0, -left), max(0, -top)
        padded = np.zeros((bottom - top, right - left, imc), dtype=np.uint8)
        padded[by:by+ch,bx:bx+cw] = cropped
        return padded

    def _crop_square_image_with_rect(self, img, rect):
        width, height = rect.width, rect.height
        cx, cy = rect.x + int(width / 2), rect.y + int(height / 2)
        width = width * 2.0
        height = height * 2.0
        left, right = cx - int(width / 2), cx + int(width / 2)
        top, bottom = cy - int(width / 2), cy + int(width / 2)
        imh, imw, imc = img.shape
        cropped = img[max(0, top):max(min(imh, bottom), 0), max(0, left):max(min(imw, right), 0)]
        ch, cw = cropped.shape[:2]
        bx, by = max(0, -left), max(0, -top)
        padded = np.zeros((bottom - top, right - left, imc), dtype=np.uint8)
        padded[by:by+ch,bx:bx+cw] = cropped
        return cropped


if __name__ == '__main__':
    rospy.init_node('people_pose_estimation_2d')
    PeoplePoseEstimation2D()
    rospy.spin()
