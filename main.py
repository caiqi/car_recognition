# -*- coding: utf-8 -*-

"""
PyQt5 tutorial

In this example, we dispay an image
on the window.

author: py40.com
last edited: 2017年3月
"""
import sys
from PyQt5.QtWidgets import (QWidget, QHBoxLayout,
                             QLabel, QApplication, QFileDialog, QMessageBox, QGridLayout, QLineEdit, QTextEdit,
                             QPushButton, QAction)
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon
import numpy as np
from PyQt5.QtCore import *
import time
import signal

from PIL import Image
import os

import utils
import json

import threading


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.topk = 5

        self.initUI()

    def initUI(self):
        self.show_log = False

        title = QPushButton('Load test', self)
        title_train = QPushButton('Load train data', self)

        title.clicked.connect(self.showDialog)
        title_train.clicked.connect(self.get_folder)

        build_data = QPushButton('Build data', self)
        start_train = QPushButton('Start training', self)
        stop_train = QPushButton('Stop training', self)

        build_data.clicked.connect(self.build_data_func)
        start_train.clicked.connect(self.start_train_func)
        stop_train.clicked.connect(self.stop_train_func)

        titleEdit = QLineEdit()
        title_train_Edit = QLineEdit()
        self.title_train_Edit = title_train_Edit
        self.titleEdit = titleEdit

        upper_grid = QGridLayout()
        upper_grid.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(10)

        grid_train = QGridLayout()
        grid_train.setSpacing(10)

        grid_train.addWidget(title_train, 1, 0)
        grid_train.addWidget(title_train_Edit, 1, 1,1,-1)
        grid_train.addWidget(build_data, 2, 0)
        grid_train.addWidget(start_train, 2, 1)
        grid_train.addWidget(stop_train, 2, 2)
        reviewEdit = QLabel('')
        reviewEdit.setFixedWidth(380)

        # reviewEdit = QTextEdit()
        reviewEdit.setWordWrap(True)

        grid_train.addWidget(reviewEdit, 3, 0, -1, 3)
        self.reviewEdit = reviewEdit

        grid.addWidget(title, 1, 0)
        grid.addWidget(titleEdit, 1, 1)
        dummpy2 = QLabel('')
        grid.addWidget(dummpy2, 2, 0, 1, -1)

        self.resize_image("blank.jpg")
        pixmap = QPixmap("load.jpg")
        pixmap = pixmap.scaledToWidth(400)
        pixmap = pixmap.scaledToHeight(400)

        lbl = QLabel(self)
        self.lbl = lbl
        lbl.setPixmap(pixmap)

        grid.addWidget(lbl, 3, 0, 1, 2)

        Results = QLabel('Results: ')

        self.output = QLabel("")
        grid.addWidget(Results, 4, 0, 1, 1)
        grid.addWidget(self.output, 4, 1, 1, 1)

        upper_grid.addLayout(grid_train, 1, 0)
        upper_grid.addLayout(grid, 1, 1)

        self.setLayout(upper_grid)

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('CNN classifier')
        self.setFixedSize(800,600)
        self.show()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        self.titleEdit.setText(fname)
        self.resize_image(fname)
        pixmap = QPixmap("load.jpg")
        pixmap = pixmap.scaledToWidth(400)
        pixmap = pixmap.scaledToHeight(400)

        self.lbl.setPixmap(pixmap)

        results = self.make_prediction()
        out_str = ""
        existing = 0

        for m, k in results:
            out_str += m + " : " + str(k) + "\n"
            existing += 1
            if existing >= self.topk:
                break
        out_str = out_str.strip()
        self.output.setText(out_str)

    def fill_log(self):
        while True:
            if not self.show_log:
                break
            time.sleep(1)
            if not os.path.exists("log/log.log"):
                continue

            content = open("log/log.log").readlines()
            if len(content) > 9:
                content = content[-9:]
            content = "\n".join(content)
            self.reviewEdit.setText(content)

    def get_folder(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open file', '/home')
        self.title_train_Edit.setText(fname)
        return

    def build_data_func(self):
        path = self.title_train_Edit.text()
        utils.build_training_data_list(path, output_classes_path="./dataset_namelist/classes.json",
                                       output_train_path="./dataset_namelist/train.txt",
                                       output_test_path="./dataset_namelist/val.txt", train_ratio=0.9)
        QMessageBox.information(self,  # 使用infomation信息框
                                    "标题",
                                    "Finishded build data")
    def start_train_func(self):
        utils.build_models(template_path="./lib/finetune_template.prototxt",
                           classes_path="./dataset_namelist/classes.json", output_model_path="./lib/finetune.prototxt")

        utils.build_models(template_path="./lib/deploy_template.prototxt",
                           classes_path="./dataset_namelist/classes.json", output_model_path="./lib/deploy.prototxt")
        self.show_log = True
        self.process_pid = utils.create_train_process()
        self.show_log_thread = threading.Thread(target=self.fill_log, name="fill")
        self.show_log_thread.start()

    def stop_train_func(self):
        self.show_log = False
        # os.killpg(os.getpgid(self.process_pid.pid), signal.SIGTERM)
        self.process_pid.kill()

    def make_prediction(self):
        model_path = os.listdir("results")
        latest = -1
        path_save = ""
        for m in model_path:
            if not m.endswith(".caffemodel"):
                continue

            if os.path.getmtime(os.path.join("results",m)) > latest:
                latest = os.path.getmtime(os.path.join("results",m))
                path_save = m

        model_path = os.path.join("results", path_save)
        results = utils.infer(self.titleEdit.text(), proto_path="./lib/deploy.prototxt",
                              model_path=model_path)
        load_classes = json.load(open("./dataset_namelist/classes.json"))
        sorted_prediction = np.argsort(-1. * results)
        sorted_results = results[sorted_prediction]
        sorted_classes = [load_classes[k] for k in sorted_prediction]
        return [[k, v] for k, v in zip(sorted_classes, sorted_results)]

    def resize_image(self, path):
        img = Image.open(path)
        width, height = img.size
        old_size = img.size

        max_dim = max(width, height)

        new_size = (max_dim, max_dim)
        new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
        new_im.paste(img, (int((new_size[0] - old_size[0]) / 2),
                           int((new_size[1] - old_size[1]) / 2)))
        img = new_im.resize((400, 400), Image.ANTIALIAS)
        img.save("load.jpg")

    def on_click(self):
        print('PyQt5 button click')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
