from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QFileDialog, QGraphicsScene

import image.icon_rc

import multiprocessing as mp

from PIL import Image
from transformers import AutoTokenizer,  GPT2Config
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTImageProcessor

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(877, 555)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 30, 861, 481))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.frame_2)
        self.graphicsView.setGeometry(QtCore.QRect(10, 20, 461, 451))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(540, 410, 111, 41))
        self.label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setScaledContents(False)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.frame_2)
        self.line.setGeometry(QtCore.QRect(630, 400, 151, 81))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.textEdit = QtWidgets.QTextEdit(self.frame_2)
        self.textEdit.setGeometry(QtCore.QRect(480, 20, 371, 351))
        self.textEdit.setObjectName("textEdit")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(490, 330, 31, 31))
        self.label_2.setStyleSheet("image: url(:/icon/icon.jpg);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(120, 10, 91, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 877, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "SCORE: "))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Select Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Select Model"))



class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.select_image)
        self.pushButton_2.clicked.connect(self.select_model)
        
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = None
    def select_image(self):
        self.options = QFileDialog.Options()
        self.options |= QFileDialog.ReadOnly  # 只讀模式
        self.img_name, _ = QFileDialog.getOpenFileName(
            filter="Image File (*.jpg *.bmp *.ppm *.jfif *.png)"
        )
        # update image
        if self.img_name != "":
            self.qpixmap = QtGui.QPixmap()
            self.qpixmap.load(self.img_name)
            self.qpixmap = self.qpixmap.scaled(480,480, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            scene = QGraphicsScene()
            scene.addPixmap(self.qpixmap)
            self.graphicsView.setScene(scene)
            print("Successfully load image !")

            if self.model is not None:
                print("Generating text... ")
                self.img = Image.open(self.img_name).convert("RGB")
                self.generated_caption = self.tokenizer.decode(self.model.generate(self.feature_extractor(self.img, return_tensors="pt").pixel_values)[0])
                # Print the generated caption in the textEdit widget
                self.textEdit.setHtml(self.generated_caption[:400])  
                # print('\033[96m' +self.generated_caption[:]+ '\033[0m')
                self.textEdit.setFont(QtGui.QFont("Times New Roman", 16))
                self.textEdit.setAlignment(QtCore.Qt.AlignJustify)
                # self.textEdit.setFontPointSize(16)
                print("Successfully generated caption !")
            else:
                print("Please select a model first.")    

    def select_model(self):
        self.options = QFileDialog.Options()
        self.options |= QFileDialog.ReadOnly
        self.model_name = QFileDialog.getExistingDirectory(self)
        if self.model_name != "":
            # 在這裡處理選擇的模型文件，您可以加載它或執行其他操作。
            self.config = VisionEncoderDecoderConfig.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name, config=self.config)
            self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
            print("Successfully load model !")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
