# coding=utf-8
import os
import sys
import cv2
import numpy as np
import math
import traceback
import time
import threading

output_label = []

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

def feature_match(img_feature, template_features, minthresh=0.3, stopthresh=0.9):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=32)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    output_label = []
    for i in range(0, len(template_features)):
        l_dict = {'object_id': template_features[i]['object_id'], 'conf':0., 'bndbox':None}
        output_label.append(l_dict)
    l_dict = {'object_id':-1, 'conf': 1, 'bndbox': None}
    output_label.append(l_dict)

    idx = 0
    for label_feature in template_features:
        for t_feature in label_feature['feature']:
            matches = matcher.knnMatch(t_feature['des'], img_feature['des'], k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            confidence = len(good) / (8 + 0.3 * len(matches))

            if len(good) >= 100 or confidence >= minthresh:
                if output_label[-1]['conf'] == 1:
                    output_label[-1]['conf'] = 0

                if confidence > output_label[idx]['conf']:
                    output_label[idx]['conf'] = confidence

                    obj = np.empty((len(good), 2), dtype=np.float32)
                    scene = np.empty((len(good), 2), dtype=np.float32)
                    for i in range(len(good)):
                        obj[i, 0] = t_feature['kp'][good[i].queryIdx].pt[0]
                        obj[i, 1] = t_feature['kp'][good[i].queryIdx].pt[1]
                        scene[i, 0] = img_feature['kp'][good[i].trainIdx].pt[0]
                        scene[i, 1] = img_feature['kp'][good[i].trainIdx].pt[1]

                    H, mask = cv2.findHomography(obj, scene, cv2.RANSAC)

                    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
                    obj_corners[0, 0, 0] = 0
                    obj_corners[0, 0, 1] = 0
                    obj_corners[1, 0, 0] = t_feature['shape'][1]
                    obj_corners[1, 0, 1] = 0
                    obj_corners[2, 0, 0] = t_feature['shape'][1]
                    obj_corners[2, 0, 1] = t_feature['shape'][0]
                    obj_corners[3, 0, 0] = 0
                    obj_corners[3, 0, 1] = t_feature['shape'][0]
                    scene_corners = cv2.perspectiveTransform(obj_corners, H)

                    scene_pnts = []
                    for j in range(0, len(scene_corners)):
                        tmp_pnt = [scene_corners[j, 0, 0], scene_corners[j, 0, 1]]
                        scene_pnts.append(tmp_pnt)

                    output_label[idx]['bndbox'] = scene_pnts

                    if output_label[idx]['conf'] >= stopthresh:
                        break

        idx += 1

    return output_label

def GetTemplateFeature(inputlist, minedge = 100, maxedge = 1000):

    templatelist = sorted(inputlist, key=lambda x:x['object_id'])
    cur_object_id = templatelist[0]['object_id']
    ids = []
    ids.append(cur_object_id)
    t_features = []
    label_feature = []
    num = 0
    for template in templatelist:
        num += 1
        if template['object_id'] != cur_object_id:
            cur_object_id = template['object_id']
            ids.append(cur_object_id)
            t_features.append(label_feature)
            label_feature = []
        image = template['img']
        if min(image.shape[0], image.shape[1]) < minedge:
            image = imresize_minedge(image, minedge)
        if max(image.shape[0], image.shape[1]) > maxedge:
            image = imresize_maxedge(image, maxedge)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        while max(gray.shape[0], gray.shape[1]) >= minedge:
            feature = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                                                              edgeThreshold=10, sigma=1.6)
            kp, des = feature.detectAndCompute(gray, None)
            f_template = {'kp': kp, 'des': des, 'shape': gray.shape}
            label_feature.append(f_template)
            gray = cv2.resize(gray, (int(gray.shape[1] * 0.5), int(gray.shape[0] * 0.5)),
                              interpolation=cv2.INTER_CUBIC)

        if num == len(templatelist):
            t_features.append(label_feature)
            label_feature = []

    template_features = []
    for i in range(0, len(t_features)):
        t_dict = {}
        t_dict['feature'] = t_features[i]
        t_dict['object_id'] = ids[i]
        template_features.append(t_dict)

    return template_features


def FindPOSM(img, template_features, minheight = 800, maxheight = 1600, minthresh=0.3, stopthresh=0.9):

    if img.shape[0] > maxheight:
        input = imresize(img, maxheight)
    else:
        input = img
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    isPOSM = False
    if gray.shape[0] < minheight:
        gray = imresize(gray, minheight)
    output_label = None
    while gray.shape[0] >= minheight:
        feature = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                                                              edgeThreshold=10, sigma=1.6)
        kp, des = feature.detectAndCompute(gray, None)
        if kp is None or len(kp) < 200:
            gray = cv2.resize(gray, (int(gray.shape[1] * 0.5), int(gray.shape[0] * 0.5)), interpolation=cv2.INTER_CUBIC)
            continue

        img_feature = {'kp': kp, 'des': des, 'shape': input.shape}
        output_label = feature_match(img_feature, template_features, minthresh=minthresh, stopthresh=stopthresh)

        if output_label[-1]['conf'] == 0:
            isPOSM = True
            for out_label in output_label:
                if out_label['object_id']!= -1 and out_label['bndbox'] != None:
                    for i in range(0, len(out_label['bndbox'])):
                        out_label['bndbox'][i] = upsample_pnts_height(gray.shape[0], input.shape[0], out_label['bndbox'][i])
                        out_label['bndbox'][i] = upsample_pnts_height(input.shape[0], img.shape[0],
                                                                      out_label['bndbox'][i])

            # for o_label in output_label:
            #     if o_label['conf'] > 0.3:
            #         for i in range(0, 4):
            #             cv2.circle(img, (o_label['bndbox'][i][0], o_label['bndbox'][i][1]), 1, (0, 255, 0), thickness=5)
            #     cv2.namedWindow("img", 2)
            #     cv2.imshow("img", img)
            #     cv2.waitKey(0)

            break

        gray = cv2.resize(gray, (int(gray.shape[1] * 0.5), int(gray.shape[0] * 0.5)), interpolation=cv2.INTER_CUBIC)

    if output_label is None:
        output_label = []
        for i in range(0, len(template_features)):
            l_dict = {'object_id': template_features[i]['object_id'], 'conf': 0., 'bndbox': None}
            output_label.append(l_dict)
        return output_label

    output_label = output_label[:-1]

    return output_label

def loop_match(label_feature, img_feature, idx, minthresh, stopthresh):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=32)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    for t_feature in label_feature['feature']:
        # t = time.time()
        matches = matcher.knnMatch(t_feature['des'], img_feature['des'], k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        confidence = len(good) / (8 + 0.3 * len(matches))
        # t = time.time() - t
        # print t_feature['shape'][0], t_feature['shape'][1], len(t_feature['kp']), \
        #     img_feature['shape'][0], img_feature['shape'][1], len(img_feature['kp']), t

        if len(good) >= 16 and (len(good) >= 100 or confidence >= minthresh):
            global output_label
            if confidence > output_label[idx]['conf']:
                output_label[idx]['conf'] = confidence

                obj = np.empty((len(good), 2), dtype=np.float32)
                scene = np.empty((len(good), 2), dtype=np.float32)
                for i in range(len(good)):
                    obj[i, 0] = t_feature['kp'][good[i].queryIdx].pt[0]
                    obj[i, 1] = t_feature['kp'][good[i].queryIdx].pt[1]
                    scene[i, 0] = img_feature['kp'][good[i].trainIdx].pt[0]
                    scene[i, 1] = img_feature['kp'][good[i].trainIdx].pt[1]

                H, mask = cv2.findHomography(obj, scene, cv2.RANSAC)

                obj_corners = np.empty((4, 1, 2), dtype=np.float32)
                obj_corners[0, 0, 0] = 0
                obj_corners[0, 0, 1] = 0
                obj_corners[1, 0, 0] = t_feature['shape'][1]
                obj_corners[1, 0, 1] = 0
                obj_corners[2, 0, 0] = t_feature['shape'][1]
                obj_corners[2, 0, 1] = t_feature['shape'][0]
                obj_corners[3, 0, 0] = 0
                obj_corners[3, 0, 1] = t_feature['shape'][0]
                scene_corners = cv2.perspectiveTransform(obj_corners, H)

                scene_pnts = []
                for j in range(0, len(scene_corners)):
                    tmp_pnt = [scene_corners[j, 0, 0], scene_corners[j, 0, 1]]
                    scene_pnts.append(tmp_pnt)

                output_label[idx]['bndbox'] = scene_pnts

                if output_label[idx]['conf'] >= stopthresh:
                    break


def feature_match_duoxiancheng(img_feature, template_features, minthresh=0.3, stopthresh=0.9, xianchengshu=1):
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    # search_params = dict(checks=32)
    # matcher = cv2.FlannBasedMatcher(index_params, search_params)

    global output_label
    output_label = []
    for i in range(0, len(template_features)):
        l_dict = {'object_id': template_features[i]['object_id'], 'conf':0., 'bndbox':None}
        output_label.append(l_dict)
    l_dict = {'object_id':-1, 'conf': 1, 'bndbox': None}
    output_label.append(l_dict)

    thread_list = []
    xianchengshu = min(xianchengshu, len(template_features))
    for idx in range(0, len(template_features)):
        thraedname = 'thread_' + str(idx)
        t1 = threading.Thread(target=loop_match, name=thraedname, args=(template_features[idx], img_feature,
                                                                       idx, minthresh, stopthresh,))
        thread_list.append(t1)

        #t = time.time()
        if (idx+1)==len(template_features):
            for tt in thread_list:
                tt.start()
            for tt in thread_list:
                tt.join()
            thread_list = []
            # t = time.time() - t
            # print t
            break

        if (idx+1)%xianchengshu == 0:
            for tt in thread_list:
                tt.start()
            for tt in thread_list:
                tt.join()
            thread_list = []
            # t = time.time() - t
            # print t

        idx += 1

    for i in range(0, len(output_label)-1):
        if output_label[i]['conf'] > 0:
            output_label[-1]['conf'] = 0

    return output_label

def FindPOSM_duoxiancheng(img, template_features, minheight = 800, maxheight = 1600, minthresh=0.3, stopthresh=0.9, xianchengshu=1):
    if img.shape[0] > maxheight:
        input = imresize(img, maxheight)
    else:
        input = img
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    isPOSM = False
    if gray.shape[0] < minheight:
        gray = imresize(gray, minheight)
    output_label = None
    while gray.shape[0] >= minheight:
        feature = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                                                              edgeThreshold=10, sigma=1.6)
        kp, des = feature.detectAndCompute(gray, None)
        if kp is None or len(kp) < 200:
            gray = cv2.resize(gray, (int(gray.shape[1] * 0.5), int(gray.shape[0] * 0.5)), interpolation=cv2.INTER_CUBIC)
            continue
        img_feature = {'kp': kp, 'des': des, 'shape': input.shape}
        output_label = feature_match_duoxiancheng(img_feature, template_features, minthresh=minthresh,
                                                  stopthresh=stopthresh, xianchengshu=xianchengshu)

        if output_label[-1]['conf'] == 0:
            isPOSM = True
            for out_label in output_label:
                if out_label['object_id']!= -1 and out_label['bndbox'] != None:
                    for i in range(0, len(out_label['bndbox'])):
                        out_label['bndbox'][i] = upsample_pnts_height(gray.shape[0], input.shape[0], out_label['bndbox'][i])
                        out_label['bndbox'][i] = upsample_pnts_height(input.shape[0], img.shape[0], out_label['bndbox'][i])

            # for o_label in output_label:
            #     if o_label['conf'] > 0.:
            #         for i in range(0, 4):
            #             cv2.circle(img, (o_label['bndbox'][i][0], o_label['bndbox'][i][1]), 1, (0, 255, 0), thickness=5)
            #     cv2.namedWindow("img", 2)
            #     cv2.imshow("img", img)
            #     cv2.waitKey(0)

            break

        gray = cv2.resize(gray, (int(gray.shape[1] * 0.5), int(gray.shape[0] * 0.5)), interpolation=cv2.INTER_CUBIC)

    if output_label is None:
        output_label = []
        for i in range(0, len(template_features)):
            l_dict = {'object_id': template_features[i]['object_id'], 'conf': 0., 'bndbox': None}
            output_label.append(l_dict)
        return output_label, False

    output_label = output_label[:-1]

    return output_label, isPOSM

def template_test(template_dir, pic_dir, output_dir):

    inputlist = []
    id = 0
    for troot, tdirs, _ in os.walk(template_dir):
        tdirs = sorted(tdirs)
        for tdir in tdirs:
            id += 1
            for _, _, tfiles in os.walk(os.path.join(troot, tdir)):
                tfiles = sorted(tfiles)
                for tfile in tfiles:
                    img = cv2.imread(os.path.join(troot, tdir, tfile))
                    t_dict = {'object_id': id, 'img': img}
                    inputlist.append(t_dict)

    template_features = GetTemplateFeature(inputlist)

    for proot, pdirs, _ in os.walk(pic_dir):
        for pdir in pdirs:
            for _, _, pfiles in os.walk(os.path.join(proot, pdir)):
                for pfile in pfiles:
                    pic = cv2.imread(os.path.join(proot, pdir, pfile))
                    output_label, isPOSM = FindPOSM_duoxiancheng(pic, template_features, xianchengshu=1)
                    for o_label in output_label:
                        if o_label['conf'] > 0.3:
                            for i in range(0, 4):
                                cv2.circle(pic, (o_label['bndbox'][i][0], o_label['bndbox'][i][1]), 1, (255, 0, 0),
                                           thickness=20)

                            for i in range(0, 3):
                                cv2.line(pic, (o_label['bndbox'][i][0], o_label['bndbox'][i][1]),\
                                         (o_label['bndbox'][i+1][0], o_label['bndbox'][i+1][1]),
                                         (0, 255, 0), thickness = 10)
                            cv2.line(pic, (o_label['bndbox'][0][0], o_label['bndbox'][0][1]), \
                                     (o_label['bndbox'][-1][0], o_label['bndbox'][-1][1]),
                                     (0, 255, 0), thickness=10)

                            # cv2.namedWindow("img", 2)
                            # cv2.imshow("img", pic)
                            # cv2.waitKey(0)

                    output_path = os.path.join(output_dir, pdir)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    output_path = os.path.join(output_path, pfile)
                    cv2.imwrite(output_path, pic)

def feature_compute(src):
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    feature = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                                          edgeThreshold=10, sigma=1.6)
    kp, des = feature.detectAndCompute(gray, None)

    if len(kp) < 10 or des is None:
        return None
    else:
        feature = {'kp': kp, 'des': des}
        return feature

def feature_matching(img_feature, template_feature):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=32)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(template_feature['des'], img_feature['des'], k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    confidence = len(good) / (8 + 0.3 * len(matches))

    return confidence

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

    _, contours, hierarchy = cv2.findContours(congray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        obj_img = gray[obj['bndbox'][0]:obj['bndbox'][2], obj['bndbox'][1]:obj['bndbox'][3]]
        # cv2.imshow('obj_img', obj_img)
        # cv2.waitKey()
        re_obj_img = imresize_maxedge(obj_img, 128)
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

    obj_list = detection(input_img)

    target_feature = feature_compute(target_img)

    match_obj_list = []
    for obj in obj_list:
        conf = feature_matching(obj['feature'], target_feature)
        if conf >= 0.5:
            match_obj_list.append(obj)

    for obj in match_obj_list:
        cv2.rectangle(input_img, (obj['bndbox'][1], obj['bndbox'][0]), (obj['bndbox'][3], obj['bndbox'][2]), (0, 255, 0), 2)

    cv2.imshow('output', input_img)
    cv2.waitKey()

    cv2.imwrite('./output.jpg',input_img)

if __name__ == '__main__':

    input_path = './source.jpg'
    target_path = './t1.jpg'

    input_img = cv2.imread(input_path)
    target_img = cv2.imread(target_path)
    # detection(input_img)
    object_detection(input_img, target_img)

    # temp1 = cv2.imread("/media/fung/jinheng/dataset/POSM/黑人/template/t1/13959/13959.jpg")
    # temp2 = cv2.imread("/home/fung/Desktop/stitch/FeatureMatchingfindObjects/POSM/黑人/template/t1/13960/13960.jpg")
    # # temp3 = cv2.imread("/1T/label/label_heiren/POSM/pics/迪丽热巴货架贴（点亮闪耀白齿，放肆亮白笑）_13960.jpg")
    # # temp4 = cv2.imread("/data/models/feature_posm/heiren0105/超白系列吊旗_13963.jpg")
    # # temp5 = cv2.imread("/data/models/feature_posm/heiren0105/迪丽热巴极尚吊旗_13962.jpg")
    # img = cv2.imread("/media/fung/jinheng/dataset/POSM/黑人/black.jpg")
    #
    # inputlist = []
    # t_dict = {'object_id': 1, 'img': temp1}
    # inputlist.append(t_dict)
    # t_dict = {'object_id': 2, 'img': temp2}
    # inputlist.append(t_dict)
    # # t_dict = {'object_id': 3, 'img': temp3}
    # # inputlist.append(t_dict)
    # # t_dict = {'object_id': 4, 'img': temp4}
    # # inputlist.append(t_dict)
    # # t_dict = {'object_id': 5, 'img': temp5}
    # # inputlist.append(t_dict)
    # template_features = GetTemplateFeature(inputlist)
    # output_label = FindPOSM(img, template_features, minthresh=0.3, stopthresh=0.9)
    # output_label = FindPOSM_duoxiancheng(img, template_features, minthresh=0.3, stopthresh=0.9, xianchengshu=2)
    # print output_label

    # for o_label in output_label:
    #     if o_label['conf'] > 0:
    #         for i in range(0, 4):
    #             cv2.circle(img, (o_label['bndbox'][i][0], o_label['bndbox'][i][1]), 1, (0,255,0), thickness=50)
    #         cv2.namedWindow("img", 2)
    #         cv2.imshow("img", img)
    #         cv2.waitKey(0)

    # t = time.time()
    # template_dir = "/media/fung/jinheng/dataset/POSM/danone/template1"
    # pic_dir = "/media/fung/jinheng/dataset/POSM/danone/test_folder"
    # output_dir = "/media/fung/jinheng/dataset/POSM/mjn/POSM14期/test_result"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # # template_test(template_dir, pic_dir, output_dir)
    # accuracy_test(template_dir, pic_dir)
    # t = time.time() - t
    # print t