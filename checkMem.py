import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QMainWindow
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from PyQt5 import QtWidgets
from LoadCam import Worker1
from threading import Thread
import detector
import os
import train
from datetime import datetime

class CheckPeople(QMainWindow):
    def __init__(self, detect=None):
        super(CheckPeople, self).__init__()
        loadUi("GUI/check.ui", self)
        self.switchAdd.clicked.connect(self.switchToAdd)
        self.exit.clicked.connect(self.exitApp)

        grid = QGridLayout()
        self.loadCam.move(70, 440)
        grid.addWidget(self.loadCam)
        self.Worker1 = Worker1()
        self.detector = detector.Detector()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.btnCheck.clicked.connect(lambda: self.cham_cong(self.Worker1.rgb))

    def cham_cong(self, im):
        thread = Thread(target=lambda: self.take_face(im))
        thread.start()


    def take_face(self, im):
        copy = im.copy()

        box, name = self.detector.detect(copy,face_only=True)

        if(box!= None and name != None):
            copy,_,_ = self.detector.get_face(copy,box)
            copy= cv2.resize(copy, (200, 200))
            FlippedImage = cv2.flip(copy, 1)
            ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0],
                                       QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(220, 220, Qt.KeepAspectRatio)
            pixmap = QPixmap(Pic)
            self.imgCheck.setPixmap(pixmap)
            self.nameCheck.setText(name)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            self.dateCheck.setText(dt_string)

    def ImageUpdateSlot(self, Image):
        self.loadCam.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

    def switchToAdd(self):
        self.Worker1.stop()
        addPeople = AddPeople()
        widget.addWidget(addPeople)
        widget.setCurrentIndex(widget.currentIndex()+1)


    def exitApp(self):
        exit(0)

class AddPeople(QMainWindow):
    def __init__(self, trainer=None):
        super(AddPeople, self).__init__()
        loadUi("GUI/add.ui", self)
        self.switchCheck.clicked.connect(self.switchToCheck)
        self.exit.clicked.connect(self.exitApp)

        self.chosen_image = None

        grid = QGridLayout()
        self.loadCam.move(70, 440)
        grid.addWidget(self.loadCam)
        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.btnCapture.clicked.connect(lambda: self.take_copy(self.Worker1.rgb))


        if (trainer==None):
            self.trainer = train.Train()
        else:
            self.trainer = trainer

    def upload(self):
        name = self.enterName.text()
        if (name != ""):
            # file = filedialog.asksaveasfilename(filetypes=[("PNG", ".png")])
            image = self.chosen_image

            # image.save(file+'.png')
            os.makedirs('Faces/' + name)
            image.save('Faces/' + name + '/' + name + '.png')
            self.addStatus.setText("Thêm thành công!")
            self.trainer.train()

    def take_copy(self, im):
        copy = im.copy()
        copy = cv2.resize(copy, (600, 360))
        FlippedImage = cv2.flip(copy, 1)
        ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0],
                                   QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(860, 520, Qt.KeepAspectRatio)
        self.chosen_image = Pic
        pixmap = QPixmap(Pic)
        self.imgCapture.setPixmap(pixmap)
        self.btnAdd.clicked.connect(self.upload)

        # self.btnAdd.clicked.connect(self.upload(self.enterName.text()))


    def ImageUpdateSlot(self, Image):
        self.loadCam.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

    def switchToCheck(self):
        checkPeople = CheckPeople()
        widget.addWidget(checkPeople)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        self.Worker1.stop()

    def exitApp(self):
        exit(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    checkPeople = CheckPeople()
    addPeople = AddPeople
    addPeople.nextFrame = CheckPeople
    widget.addWidget(checkPeople)
    widget.setFixedWidth(1440)
    widget.setFixedHeight(980)
    widget.show()
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")