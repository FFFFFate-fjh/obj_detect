# coding=utf-8
import os
import sys
import cv2
import numpy as np
import math
import traceback
import time
import threading


eps = 10e-6

def imresize(src, height):
    ratio = src.shape[0] * 1.0 / height
    width = int(src.shape[1] * 1.0 / ratio)
    return cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)

def imresize_width(src, width):
    ratio = src.shape[1] * 1.0 / width
    height = int(src.shape[0] * 1.0 / ratio)
    return cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)

def imresize_maxedge(src, maxedge):
    ratio = max(src.shape[0], src.shape[1]) * 1.0 / maxedge
    height = int(src.shape[0] * 1.0 / ratio)
    width = int(src.shape[1] * 1.0 / ratio)
    return cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)

def imresize_minedge(src, minedge):
    ratio = min(src.shape[0], src.shape[1]) * 1.0 / minedge
    height = int(src.shape[0] * 1.0 / ratio)
    width = int(src.shape[1] * 1.0 / ratio)
    return cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)

def upsample_pnts_height(resize_h, ori_h, pnts):
    ratio_h = ori_h*1./resize_h
    pnts[0] = int(pnts[0] * ratio_h)
    pnts[1] = int(pnts[1] * ratio_h)
    return pnts

def feature_compute(src):
    if len(src.shape) == 3:
        sub_imgs = cv2.split(src)
        feature = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=10e-3,
                                              edgeThreshold=100, sigma=1.6)
        kp_list = []
        des_arr = None
        for s_img in sub_imgs:
            kp, des = feature.detectAndCompute(s_img, None)
            kp_list.extend(kp)
            if des_arr is None:
                des_arr = des
            else:
                des_arr = np.vstack((des_arr, des))

        if len(kp_list) < 10 or des_arr is None:
            return None
        else:
            feature = {'kp': kp_list, 'des': des_arr}
            return feature

        # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # feature = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=10e-3,
        #                                       edgeThreshold=100, sigma=1.6)
        # kp, des = feature.detectAndCompute(gray, None)
        #
        # if len(kp) < 10 or des is None:
        #     return None
        # else:
        #     feature = {'kp': kp, 'des': des}
        #     return feature

    else:
        gray = src
        feature = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=10e-3,
                                              edgeThreshold=100, sigma=1.6)
        kp, des = feature.detectAndCompute(gray, None)

        if len(kp) < 10 or des is None:
            return None
        else:
            feature = {'kp': kp, 'des': des}
            return feature

def feature_matching(img_feature, template_feature):
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    # search_params = dict(checks=32)
    # matcher = cv2.FlannBasedMatcher(index_params, search_params)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(template_feature['des'], img_feature['des'], k=2)

    # matches = matcher.knnMatch(template_feature['des'], img_feature['des'], k=2)
    good = []
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)
            matche_pair = [m, n]
            good_matches.append(matche_pair)

    # matches2 = matcher.knnMatch(img_feature['des'], template_feature['des'], k=2)
    # matches2 = bf.knnMatch(template_feature['des'], img_feature['des'], k=2)
    # num = 0
    # for m, n in matches2:
    #     repeat = False
    #     for i in range(0, len(matches)):
    #         if m in matches[i] or n in matches[i]:
    #             repeat = True
    #             break
    #     if repeat is True:
    #         continue
    #     if m.distance < 0.9 * n.distance:
    #         num += 1
    #         good.append(m)
    #         matche_pair = [m, n]
    #         good_matches.append(matche_pair)

    confidence = len(good) / (8 + 0.3 * len(matches))
    # confidence = len(good_matches)*1. / (len(matches))
    # confidence = len(good) / (8 + 0.3 * (len(matches)+len(matches2)))

    return confidence, good_matches

# 计算单通道的直方图的相似值
def calculate(image1, image2):
    retval, image1_otsu = cv2.threshold(image1, 0, 255, cv2.THRESH_OTSU)
    retval, image2_otsu = cv2.threshold(image2, 0, 255, cv2.THRESH_OTSU)

    hist1 = cv2.calcHist([image1], [0], image1_otsu, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], image2_otsu, [256], [0.0, 255.0])

    # hist1 = cv2.normalize(hist1,  0, 1, cv2.NORM_MINMAX)
    # hist2 = cv2.normalize(hist2, 0, 1, cv2.NORM_MINMAX)

    # similarity = cv2.compareHist(hist1, hist2, 0)
    # for i in range(0, 10):
    #     hist1[i] = 0
    #     hist2[i] = 0
    #计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def classify_hist_with_split(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    conf_list = []
    for im1, im2 in zip(sub_image1, sub_image2):
        conf = calculate(im1, im2)
        sub_data += conf
        conf_list.append(float(conf))
    sub_data = sub_data / 3
    conf_arr = np.array(conf_list)
    return sub_data, conf_arr

def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 感知哈希算法(pHash)
def pHash(img):
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def template_matching(obj_img, target_img):
    # re_obj_img = imresize_maxedge(obj_img, 64)
    # re_target_img = imresize_maxedge(target_img, 64)
    re_obj_img = cv2.resize(obj_img, (64, 64), interpolation=cv2.INTER_CUBIC)
    re_target_img = cv2.resize(target_img, (64, 64), interpolation=cv2.INTER_CUBIC)
    res = cv2.matchTemplate(re_obj_img, re_target_img, cv2.TM_CCOEFF_NORMED)

    return res

def detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regray = imresize_maxedge(gray, 640)
    reimg = imresize_maxedge(img, 640)

    img_median = cv2.medianBlur(regray, 3)
    img_guassian = cv2.GaussianBlur(img_median, (3, 3), 0.5)

    retval, otsu = cv2.threshold(img_guassian, 0, 255, cv2.THRESH_OTSU)

    # open = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, (3, 3))
    # close = cv2.morphologyEx(open, cv2.MORPH_OPEN, (3, 3))

    congray = imresize_maxedge(otsu, max(img.shape))

    contours, hierarchy = cv2.findContours(congray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    obj_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img.shape[0]*img.shape[1]*10e-5:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        bbox = [y, x, y+h, x+w]
        obj_item = {'bndbox': bbox}
        obj_list.append(obj_item)

    fea_obj_list = []
    for obj in obj_list:
        obj_img = img[obj['bndbox'][0]:obj['bndbox'][2], obj['bndbox'][1]:obj['bndbox'][3]]
        # cv2.imshow('obj_img', obj_img)
        # cv2.waitKey()
        if max(obj_img.shape) < 64:
            re_obj_img = imresize_maxedge(obj_img, 64)
        else:
            re_obj_img = obj_img
        feature = feature_compute(re_obj_img)
        if feature is not None:
            new_obj = {}
            new_obj['feature'] = feature
            new_obj['bndbox'] = obj['bndbox']
            fea_obj_list.append(new_obj)
    obj_list = fea_obj_list

    # for obj in obj_list:
    #     cv2.rectangle(img, (obj['bndbox'][1], obj['bndbox'][0]), (obj['bndbox'][3], obj['bndbox'][2]), (255, 0, 0), 1)

    # cv2.imshow('test', img)
    # cv2.waitKey()

    return obj_list

def object_detection(input_img, target_img):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    obj_list = detection(input_img)

    if max(target_img.shape) < 64:
        re_target_img = imresize_maxedge(target_img, 64)
    else:
        re_target_img = target_img
    re_target_gray = cv2.cvtColor(re_target_img, cv2.COLOR_BGR2GRAY)
    target_feature = feature_compute(re_target_img)

    match_obj_list = []
    for obj in obj_list:

        obj_img = input_img[obj['bndbox'][0]:obj['bndbox'][2], obj['bndbox'][1]:obj['bndbox'][3]]
        hist_conf, hist_conf_arr = classify_hist_with_split(obj_img, re_target_img, size=(64, 64))
        if hist_conf < 0.6 or np.std(hist_conf_arr) > 0.1:
            continue
        # print hist_conf, hist_conf_arr, np.std(hist_conf_arr)
        # if hist_conf >= 0.6 and np.std(hist_conf_arr) < 0.05:
        #     match_obj_list.append(obj)
        #     continue

        feature_conf, matches = feature_matching(obj['feature'], target_feature)
        # obj_img = input_img[obj['bndbox'][0]:obj['bndbox'][2], obj['bndbox'][1]:obj['bndbox'][3]]
        # show_img = cv2.drawMatchesKnn(obj_img, obj['feature']['kp'], re_target_img, target_feature['kp'], matches, None, flags=2)
        # print feature_conf, len(matches)
        # cv2.namedWindow('match', 2)
        # cv2.imshow("match", show_img)
        # cv2.waitKey()
        if feature_conf >= 1 or len(matches) >= 60:
            match_obj_list.append(obj)
            continue

        obj_img = input_img[obj['bndbox'][0]:obj['bndbox'][2], obj['bndbox'][1]:obj['bndbox'][3]]

        obj_hash = aHash(obj_img)
        tar_hash = aHash(re_target_img)
        conf = cmpHash(obj_hash, tar_hash)
        if conf <= 10:
            match_obj_list.append(obj)
            continue

        obj_img = gray[obj['bndbox'][0]:obj['bndbox'][2], obj['bndbox'][1]:obj['bndbox'][3]]
        conf = template_matching(obj_img, re_target_gray)
        if conf >= 0.7:
            match_obj_list.append(obj)
            continue

    for obj in match_obj_list:
        cv2.rectangle(input_img, (obj['bndbox'][1], obj['bndbox'][0]), (obj['bndbox'][3], obj['bndbox'][2]), (0, 255, 0), 2)
        print obj['bndbox'][0], obj['bndbox'][1], obj['bndbox'][2], obj['bndbox'][3]

    # # cv2.namedWindow('output', 2)
    # cv2.imshow('output', input_img)
    # cv2.waitKey()

    cv2.imwrite('./output.jpg',input_img)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'need 2 input parameters'
        exit(1)

    input_path = sys.argv[1]
    target_path = sys.argv[2]

    input_img = cv2.imread(input_path)
    target_img = cv2.imread(target_path)
    # detection(input_img)
    object_detection(input_img, target_img)
