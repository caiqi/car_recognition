import os
import numpy as np
import json
import subprocess
import sys
sys.path.insert(0, r'caffe/python/')
import cv2
import caffe

def build_training_data_list(base_dir, output_classes_path, output_train_path, output_test_path, train_ratio):
    all_folders = os.listdir(base_dir)
    files_mapping = {}
    for k in all_folders:
        if k == "." or k == "..":
            continue
        files = os.listdir(os.path.join(base_dir, k))
        files = [os.path.join(base_dir, k, m) for m in files]
        files_mapping[k] = files
    all_data = []
    classes = [k for k in files_mapping.keys()]
    json.dump(classes, open(output_classes_path, "w"))

    for k, v in files_mapping.items():
        all_data = all_data + [[j, classes.index(k)] for j in v]
    np.random.shuffle(all_data)
    if train_ratio < 1.:
        train_data = all_data[:int(len(all_data) * train_ratio)]
        test_data = all_data[int(len(all_data) * train_ratio):]
        with open(output_train_path, "w") as g:
            for m in train_data:
                g.write(m[0] + " " + str(m[1]) + "\n")

        with open(output_test_path, "w") as g:
            for m in test_data:
                g.write(m[0] + " " + str(m[1]) + "\n")
    else:
        train_data = all_data
        with open(output_train_path, "w") as g:
            for m in train_data:
                g.write(m[0] + " " + str(m[1]) + "\n")

def build_models(template_path, classes_path, output_model_path):
    data = json.load(open(classes_path))
    class_num = len(data)
    content = open(template_path).read().replace("$$$$", str(class_num))
    with open(output_model_path, "w") as g:
        g.write(content)
    return

def create_train_process():
    log = open("log/log.log", "w", 1)
    process = subprocess.Popen([r"caffe\bin\caffe.exe", "train", "--solver=./lib/solver_finetune.prototxt", "--weights=./ImageNet_pretrained_model/squeezenet_v1.1.caffemodel"],cwd=os.path.dirname(os.path.realpath(__file__)), stderr =log)
    return process

def load_image_for_pretrained_caffemodel(imPath):
    im = cv2.imread(imPath)
    if len(im.shape) == 2:
        new_im = np.zeros((im.shape[0], im.shape[1], 3))
        new_im[:, :, 0] = im
        new_im[:, :, 1] = im
        new_im[:, :, 2] = im
        im = new_im
    im = cv2.resize(im, (227, 227), interpolation=cv2.INTER_LINEAR)
    mean = np.array([104, 117, 123])
    im = im - mean
    im = im.transpose((2, 0, 1))  # HWC -> CHW
    return im

def infer(imPath,proto_path, model_path):
    caffe.set_mode_cpu()
    deploy_prototxt = proto_path
    caffe_model = model_path
    net = caffe.Net(deploy_prototxt, caffe_model, caffe.TEST)
    testIm = load_image_for_pretrained_caffemodel(imPath)
    net.blobs['data'].reshape(1, 3, 227, 227)
    net.blobs['data'].data[0, :, :, :] = testIm
    net.forward()
    testIm_feature = net.blobs['prob'].data.copy()
    testIm_feature = testIm_feature.squeeze()
    return testIm_feature
