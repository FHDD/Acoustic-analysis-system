# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'choose.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 278)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(70, 113, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(220, 113, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(70, 190, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(220, 190, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(150, 40, 81, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.ecgbutton_click)
        self.pushButton_2.clicked.connect(Form.ecgbutton_clicktwo)
        self.pushButton_3.clicked.connect(Form.ecgbutton_clickthree)
        self.pushButton_4.clicked.connect(Form.ecgbutton_clickfour)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "function"))
        self.pushButton.setText(_translate("Form", "录音分析"))
        self.pushButton_2.setText(_translate("Form", "参数提取"))
        self.pushButton_3.setText(_translate("Form", "语音识别"))
        self.pushButton_4.setText(_translate("Form", "声转文本"))
        self.label.setText(_translate("Form", "功能模块"))
