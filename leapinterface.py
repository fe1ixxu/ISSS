# -*- coding: utf-8 -*-

import Leap,sys
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QFont
from PyQt5.QtGui import (QPainter, QPen)
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from PyQt5 import QtGui
import time
import numpy as np
import fastdtw
import MySQLdb
import heapq
import numpy
from ast import literal_eval
from collections import Counter
dataset=[]
Person=[]
Text=0
pos_xy=[]
STOPFLAG=False
DIS=1
speeda=[]
speedb=[]
speedc=[]
KNN_K=3
Hand_type=True
Calstep=0
ST=0
STOPGESTURE=False
# tdis=[6.314,3.220,2.753,2.132,2.015,1.943,1.895,1.860,1.833,1.812,1.796,1.782,1.771,1.761]
tablename=['signature', 'signature1', 'signature2', 'signature3', 'signature4', 'signature5', 'signature6', 'signature7', 'signature8', 'signature9', 'signature10', 'signature11', 'signature12', 'signature13', 'signature14', 'signature15', 'signature16', 'signature17', 'signature18', 'signature19', 'signature20', 'signature21', 'signature22', 'signature23', 'signature24', 'signature25', 'signature26', 'signature27', 'signature28', 'signature29', 'signature30', 'signature31', 'signature32', 'signature33', 'signature34', 'signature35', 'signature36', 'signature37', 'signature38', 'signature39', 'signature40', 'signature41', 'signature42', 'signature43', 'signature44', 'signature45', 'signature46', 'signature47', 'signature48', 'signature49', 'signature50', 'signature51', 'signature52', 'signature53', 'signature54', 'signature55', 'signature56', 'signature57', 'signature58', 'signature59', 'signature60', 'signature61', 'signature62', 'signature63', 'signature64', 'signature65', 'signature66', 'signature67', 'signature68', 'signature69', 'signature70', 'signature71', 'signature72', 'signature73', 'signature74', 'signature75', 'signature76', 'signature77', 'signature78', 'signature79', 'signature80', 'signature81', 'signature82', 'signature83', 'signature84', 'signature85', 'signature86', 'signature87', 'signature88', 'signature89', 'signature90', 'signature91', 'signature92', 'signature93', 'signature94', 'signature95', 'signature96', 'signature97', 'signature98', 'signature99', 'signature100', 'signature101', 'signature102', 'signature103', 'signature104', 'signature105', 'signature106', 'signature107', 'signature108', 'signature109', 'signature110', 'signature111', 'signature112', 'signature113', 'signature114', 'signature115', 'signature116', 'signature117', 'signature118', 'signature119', 'signature120', 'signature121', 'signature122', 'signature123', 'signature124', 'signature125', 'signature126', 'signature127', 'signature128', 'signature129', 'signature130', 'signature131', 'signature132', 'signature133', 'signature134', 'signature135', 'signature136', 'signature137', 'signature138', 'signature139', 'signature140', 'signature141', 'signature142', 'signature143', 'signature144', 'signature145', 'signature146', 'signature147', 'signature148', 'signature149', 'signature150', 'signature151', 'signature152', 'signature153', 'signature154', 'signature155', 'signature156', 'signature157', 'signature158', 'signature159', 'signature160', 'signature161', 'signature162', 'signature163', 'signature164', 'signature165', 'signature166', 'signature167', 'signature168', 'signature169', 'signature170', 'signature171', 'signature172', 'signature173', 'signature174', 'signature175', 'signature176', 'signature177', 'signature178', 'signature179', 'signature180', 'signature181', 'signature182', 'signature183', 'signature184', 'signature185', 'signature186', 'signature187', 'signature188', 'signature189', 'signature190', 'signature191', 'signature192', 'signature193', 'signature194', 'signature195', 'signature196', 'signature197', 'signature198', 'signature199', 'signature200', 'signature201', 'signature202', 'signature203', 'signature204', 'signature205', 'signature206', 'signature207', 'signature208', 'signature209', 'signature210', 'signature211', 'signature212', 'signature213', 'signature214', 'signature215', 'signature216', 'signature217', 'signature218', 'signature219', 'signature220', 'signature221', 'signature222', 'signature223', 'signature224', 'signature225', 'signature226', 'signature227', 'signature228', 'signature229', 'signature230', 'signature231', 'signature232', 'signature233', 'signature234', 'signature235', 'signature236', 'signature237', 'signature238', 'signature239', 'signature240', 'signature241', 'signature242', 'signature243', 'signature244', 'signature245', 'signature246', 'signature247', 'signature248', 'signature249', 'signature250', 'signature251', 'signature252', 'signature253', 'signature254', 'signature255', 'signature256', 'signature257', 'signature258', 'signature259', 'signature260', 'signature261', 'signature262', 'signature263', 'signature264', 'signature265', 'signature266', 'signature267', 'signature268', 'signature269', 'signature270', 'signature271', 'signature272', 'signature273', 'signature274', 'signature275', 'signature276', 'signature277', 'signature278', 'signature279', 'signature280', 'signature281', 'signature282', 'signature283', 'signature284', 'signature285', 'signature286', 'signature287', 'signature288', 'signature289', 'signature290', 'signature291', 'signature292', 'signature293', 'signature294', 'signature295', 'signature296', 'signature297', 'signature298', 'signature299', 'signature300', 'signature301', 'signature302', 'signature303', 'signature304', 'signature305', 'signature306', 'signature307', 'signature308', 'signature309', 'signature310', 'signature311', 'signature312', 'signature313', 'signature314', 'signature315', 'signature316', 'signature317', 'signature318', 'signature319', 'signature320', 'signature321', 'signature322', 'signature323', 'signature324', 'signature325', 'signature326', 'signature327', 'signature328', 'signature329', 'signature330', 'signature331', 'signature332', 'signature333', 'signature334', 'signature335', 'signature336', 'signature337', 'signature338', 'signature339', 'signature340', 'signature341', 'signature342', 'signature343', 'signature344', 'signature345', 'signature346', 'signature347', 'signature348', 'signature349', 'signature350', 'signature351', 'signature352', 'signature353', 'signature354', 'signature355', 'signature356', 'signature357', 'signature358', 'signature359', 'signature360', 'signature361', 'signature362', 'signature363', 'signature364', 'signature365', 'signature366', 'signature367', 'signature368', 'signature369', 'signature370', 'signature371', 'signature372', 'signature373', 'signature374', 'signature375', 'signature376', 'signature377', 'signature378', 'signature379', 'signature380', 'signature381', 'signature382', 'signature383', 'signature384', 'signature385', 'signature386', 'signature387', 'signature388', 'signature389', 'signature390', 'signature391', 'signature392', 'signature393', 'signature394', 'signature395', 'signature396', 'signature397', 'signature398', 'signature399', 'signature400', 'signature401', 'signature402', 'signature403', 'signature404', 'signature405', 'signature406', 'signature407', 'signature408', 'signature409', 'signature410', 'signature411', 'signature412', 'signature413', 'signature414', 'signature415', 'signature416', 'signature417', 'signature418', 'signature419', 'signature420', 'signature421', 'signature422', 'signature423', 'signature424', 'signature425', 'signature426', 'signature427', 'signature428', 'signature429', 'signature430', 'signature431', 'signature432', 'signature433', 'signature434', 'signature435', 'signature436', 'signature437', 'signature438', 'signature439', 'signature440', 'signature441', 'signature442', 'signature443', 'signature444', 'signature445', 'signature446', 'signature447', 'signature448', 'signature449', 'signature450', 'signature451', 'signature452', 'signature453', 'signature454', 'signature455', 'signature456', 'signature457', 'signature458', 'signature459', 'signature460', 'signature461', 'signature462', 'signature463', 'signature464', 'signature465', 'signature466', 'signature467', 'signature468', 'signature469', 'signature470', 'signature471', 'signature472', 'signature473', 'signature474', 'signature475', 'signature476', 'signature477', 'signature478', 'signature479', 'signature480', 'signature481', 'signature482', 'signature483', 'signature484', 'signature485', 'signature486', 'signature487', 'signature488', 'signature489', 'signature490', 'signature491', 'signature492', 'signature493', 'signature494', 'signature495', 'signature496', 'signature497', 'signature498', 'signature499', 'signature500', 'signature501', 'signature502', 'signature503', 'signature504', 'signature505', 'signature506', 'signature507', 'signature508', 'signature509', 'signature510', 'signature511', 'signature512', 'signature513', 'signature514', 'signature515', 'signature516', 'signature517', 'signature518', 'signature519', 'signature520', 'signature521', 'signature522', 'signature523', 'signature524', 'signature525', 'signature526', 'signature527', 'signature528', 'signature529', 'signature530', 'signature531', 'signature532', 'signature533', 'signature534', 'signature535', 'signature536', 'signature537', 'signature538', 'signature539', 'signature540', 'signature541', 'signature542', 'signature543', 'signature544', 'signature545', 'signature546', 'signature547', 'signature548', 'signature549', 'signature550', 'signature551', 'signature552', 'signature553', 'signature554', 'signature555', 'signature556', 'signature557', 'signature558', 'signature559', 'signature560', 'signature561', 'signature562', 'signature563', 'signature564', 'signature565', 'signature566', 'signature567', 'signature568', 'signature569', 'signature570', 'signature571', 'signature572', 'signature573', 'signature574', 'signature575', 'signature576', 'signature577', 'signature578', 'signature579', 'signature580', 'signature581', 'signature582', 'signature583', 'signature584', 'signature585', 'signature586', 'signature587', 'signature588', 'signature589', 'signature590', 'signature591', 'signature592', 'signature593', 'signature594', 'signature595', 'signature596', 'signature597', 'signature598', 'signature599', 'signature600']
class MainWindow(QWidget):


    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):

        self.trainbutton= QPushButton('Train', self)
        self.trainbutton.setFont(QFont("Marker Felt",40,QFont.Black))
        self.trainbutton.resize(300, 200)
        self.trainbutton.move(880,100)


        self.testbutton=QPushButton('Test',self)
        self.testbutton.resize(300,200)
        self.testbutton.move(880,400)
        self.testbutton.setFont(QFont("Marker Felt",40,QFont.Black))

        quitbutton=QPushButton('Quit',self)
        quitbutton.resize(300,200)
        quitbutton.move(880,700)
        quitbutton.clicked.connect(self.quit)
        quitbutton.setFont(QFont("Marker Felt",40,QFont.Black))
        quitbutton.setShortcut('Ctrl+Q')

        ql1=QLabel('ISSS designer:',self)
        ql1.move(40,40)
        ql1.setFont(QFont("Marker Felt",20,QFont.Black))
        ql2=QLabel('Haoran Xu',self)
        ql2.move(40,80)
        ql2.setFont(QFont("Marker Felt",20,QFont.Black))
        # ql3=QLabel('Feiwu Wang',self)
        # ql3.move(40,120)
        # ql3.setFont(QFont("Marker Felt",20,QFont.Black))
        # ql4=QLabel('Liming Bao',self)
        # ql4.move(40,160)
        # ql4.setFont(QFont("Marker Felt",20,QFont.Black))
        # ql5=QLabel('Feng Chen',self)
        # ql5.move(40,200)
        # ql5.setFont(QFont("Marker Felt",20,QFont.Black))
        image=QLabel(self)
        image.move(1500,100)
        picture = QtGui.QPixmap('D:/ISSD_image/isss.png')
        image.setPixmap(picture)
        image.show()

        self.resize(1940,1040)
        self.center()
        self.setWindowTitle('Invisible Signature Security System')
        self.setWindowIcon(QIcon('web.png'))

        cb = QCheckBox('Are you left-handed?', self)
        cb.move(50, 800)
        cb.stateChanged.connect(self.changeHandtype)
        cb.setFont(QFont("Marker Felt",30,QFont.Black))
        cb.adjustSize()

        self.show()
    def changeHandtype(self,state):
        global Hand_type
        if state == Qt.Checked:
            Hand_type=False
        else:
            Hand_type=True




    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def buttoncenter(self,button):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        button.move(qr.topLeft())

    def quit(self):
        self.close()

class DrawWindow(QWidget):

    def __init__(self):
        super(DrawWindow, self).__init__(parent=None)
        self.listener = SampleListener()
        self.controller = Leap.Controller()
        self.resize(1940, 1080)
        self.move(0, 0)
        self.setWindowTitle("ISSD")

        qbtn = QPushButton('Train', self)
        qbtn.clicked.connect(self.buttonClicked)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(0, 0)
        qbtn.setShortcut(Qt.Key_Return)

        end=QPushButton('End',self)
        end.clicked.connect(self.end)
        end.resize(end.sizeHint())
        end.move(0,25)
        end.setShortcut('Esc')

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(700,650,400,25)
        self.pbar.close()
        self.timer = QBasicTimer()

        self.cal=QLabel('Recording....',self)
        self.cal.setFont(QFont("Marker Felt", 10, QFont.Black))
        self.cal.move(850,680)
        self.cal.close()

        self.lb1 = QLabel('              When you move your finger, a black point will follow it\n                            If you are ready to sign\n Please keep your finger STILL for 2 or 3 seconds until the black frame becomes to RED',self)
        self.lb2 = QLabel('              When you sign your name, try your best to keep it in the red frame\nIf you finished, keep your finger STILL for 2 or 3 seconds untill the red frame becomes to BLACK AGAIN',self)
        self.lb3 = QLabel('All done! just click ‘Train’button or press Enter to record your name!',self)


    def buttonClicked(self):
        global pos_xy
        global dataset
        global Person
        global Text
        global STOPFLAG
        global DIS
        global speeda,speedb,speedc
        global ST
        if pos_xy==[(-1,-1,-1)] or len(pos_xy)==2:
            QMessageBox.information(self,"Warning", "Please input your name!")
            return
        while (-1,-1,-1) in pos_xy:
            pos_xy.remove((-1,-1,-1))
        pos_xy=pos_xy[::2]
        # pos_xy=pos_xy[::2]
        Csa,Csb,Csr=Direction(pos_xy)
        Sa,Sb,Sc=Slope(pos_xy)
        # A1,A2,A3=Aspect(pos_xy)
        acc1,acc2,acc3=Acceleration(speeda,speedb,speedc)
        self.Signature=zip(Csa,Csb,Csr,Sa,Sb,Sc,speeda,speedb,speedc,acc1,acc2,acc3)
        db=self.Connectmysql()
        self.Insertmysql(db)
        self.timer.stop()
        ST=0
        self.Closemysql(db)
        self.pbar.close()
        self.cal.close()
        pos_xy=[]
        SpeedClear()
        self.update()
        STOPFLAG=False
        DIS=1



    def paintEvent(self, event):
        global pos_xy
        global DIS
        global STOPFLAG
        self.controller.add_listener(self.listener)
        painter = QPainter()
        painter.begin(self)

        if (DIS==1 and STOPFLAG==False):
            pen = QPen(Qt.black, 6, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(500,200,740,400)
            self.lb3.close()
            self.lb2.close()
            self.lb1.show()
            self.lb1.move(245, 10)
            self.lb1.setFont(QFont("Marker Felt", 20, QFont.Black))
            self.lb1.adjustSize()
        elif DIS==-1 and STOPFLAG==True:
            pen = QPen(Qt.black, 6, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(500,200,740,400)
            self.lb2.close()
            self.lb3.show()
            self.lb3.move(400,35)
            self.lb3.setFont(QFont("Marker Felt", 20, QFont.Black))
            self.lb3.adjustSize()
        else:
            pen = QPen(Qt.red, 6, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(500,200,740,400)
            self.lb2.show()
            self.lb1.close()
            self.lb2.move(245,35)
            self.lb2.setFont(QFont("Marker Felt", 20, QFont.Black))
            self.lb2.adjustSize()


        pen = QPen(Qt.black, 6, Qt.SolidLine)
        painter.setPen(pen)


        if len(pos_xy) > 1:
            point_start = pos_xy[0]
            for pos_tmp in pos_xy:
                point_end = pos_tmp
                if point_end == (-1, -1,-1):
                    point_start = (-1, -1,-1)
                    continue
                if point_start == (-1, -1,-1):
                    point_start = point_end
                    continue
                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()
        self.update()
    def Connectmysql(self):
        print 'connecting to database......'
        conn = MySQLdb.connect(
            host='localhost',
            port=3306,
            user='root',

            passwd='qweruiop',
            db='test',
            charset='utf8',
        )
        print 'connected'
        return conn

    def Closemysql(self,db):
        db.close()
        print "closed"
    def Insertmysql(self,db):
        global ST
        cursor = db.cursor()
        idnum=self.getidnum(db)
        print "start inserting.........."
        chu=5
        self.tablenum=int(len(self.Signature) / chu )+ 1
        if self.tablenum>600:
            QMessageBox.information(self, "Warning", 'Name is too long!')
            return
        self.pbar.setMaximum(self.tablenum)
        self.pbar.show()
        self.cal.show()
        self.timer.start(self.tablenum, self)
        self.update()
        print self.tablenum
        for i in range(self.tablenum):
            data = self.Signature[ i*chu:(i + 1) * chu]
            if i == 0:
                sql = "insert into %s(id,Person,Dataset) values ('%d','%s','%s')" % (tablename[i], idnum, Text, data)
                cursor.execute(sql)
                db.commit()
            elif i == 1:
                sql = "insert into %s(id,Dataset,tablenum) values ('%d','%s','%d')" % (tablename[i], idnum, data,self.tablenum)
                cursor.execute(sql)
                db.commit()
            else:
                sql = "insert into %s(id,Dataset) values ('%d','%s')" % (tablename[i], idnum, data)
                cursor.execute(sql)
                db.commit()
            ST=ST+1
            self.pbar.setValue(ST)




        cursor.close()
        print "insert successfull!"

    def getidnum(self,db):
        cursor=db.cursor()
        sql="select * from signature"
        cursor.execute(sql)
        db.commit()
        idnum=cursor.rowcount
        cursor.close()
        return idnum
    def end(self):
        global pos_xy
        global STOPFLAG
        global DIS
        self.controller.remove_listener(self.listener)
        pos_xy=[]
        STOPFLAG=False
        DIS=1
        SpeedClear()
        self.close()

    def display(self):
        self.show()


class TestWindow(QWidget):
    def __init__(self):
        global DIS
        global STOPFLAG
        super(TestWindow, self).__init__(parent=None)
        self.listener=SampleListener()
        self.controller=Leap.Controller()
        self.resize(1940, 1080)
        self.move(0, 0)
        self.setWindowTitle("ISSD")
        self.distance=[]
        self.signature=[]
        self.minindex=[]
        self.rec=True

        self.qbtn = QPushButton('Test', self)
        self.qbtn.clicked.connect(self.buttonClicked)
        self.qbtn.resize(self.qbtn.sizeHint())
        self.qbtn.move(0, 0)
        self.qbtn.setShortcut(Qt.Key_Return)

        self.qbtn2 = QPushButton('Restart', self)
        self.qbtn2.clicked.connect(self.buttonClicked2)
        self.qbtn2.resize(self.qbtn.sizeHint())
        self.qbtn2.move(0, 0)
        self.qbtn2.setShortcut(Qt.Key_Return)
        self.qbtn2.close()

        db=self.Connectmysql()
        self.maxnum=self.getrownum(db)
        self.Closemysql(db)

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(700,650,400, 25)
        self.pbar.setMaximum(self.maxnum+3)
        self.pbar.close()
        self.timer = QBasicTimer()

        self.cal=QLabel('Calculating....',self)
        self.cal.setFont(QFont("Marker Felt", 10, QFont.Black))
        self.cal.move(850,680)
        self.cal.close()

        self.showname=QLabel(self)
        self.showname.setFont(QFont("Marker Felt", 70, QFont.Black))
        self.showname.move(450,325)
        self.showname.close()

        self.similarity=QLabel(self)
        self.similarity.setFont(QFont("Marker Felt", 25, QFont.Black))
        self.similarity.move(450,475)
        self.similarity.close()

        self.log=QLabel("Login successfully!",self)
        self.log.setFont(QFont("Marker Felt", 35, QFont.Black))
        self.log.move(450,200)
        self.log.close()

        end=QPushButton('End',self)
        end.clicked.connect(self.end)
        end.resize(end.sizeHint())
        end.move(0,25)
        end.setShortcut('Esc')

        self.image=QLabel(self)
        self.image.move(1200, 200)
        self.image.resize(640,480)




        self.lb1 = QLabel('              When you move your finger, a black point will follow it\n                            If you are ready to sign\n Please keep your finger STILL for 2 or 3 seconds until the black frame becomes to RED',self)
        self.lb2 = QLabel('              When you sign your name, try your best to keep it in the red frame\nIf you finished, keep your finger STILL for 2 or 3 seconds untill the red frame becomes to BLACK AGAIN',self)
        self.lb3 = QLabel('All done! just click ‘Test’button or press Enter to recognize your name!',self)



    def paintEvent(self, event):
        global pos_xy
        global DIS
        global STOPFLAG
        self.controller.add_listener(self.listener)
        painter = QPainter()
        painter.begin(self)


        if (DIS==1 and STOPFLAG==False):
            pen = QPen(Qt.black, 6, Qt.SolidLine)
            painter.setPen(pen)
            if self.rec==True:
                painter.drawRect(500,200,740,400)
            self.lb3.close()
            self.lb2.close()
            self.lb1.show()
            self.lb1.move(245, 10)
            self.lb1.setFont(QFont("Marker Felt", 20, QFont.Black))
            self.lb1.adjustSize()
        elif DIS==-1 and STOPFLAG==True:
            pen = QPen(Qt.black, 6, Qt.SolidLine)
            painter.setPen(pen)
            self.lb2.close()
            if self.rec==True:
                painter.drawRect(500,200,740,400)
                self.lb3.show()
                self.lb3.move(400, 35)
                self.lb3.setFont(QFont("Marker Felt", 20, QFont.Black))
                self.lb3.adjustSize()
            else:
                self.lb3.close()


        else:
            pen = QPen(Qt.red, 6, Qt.SolidLine)
            painter.setPen(pen)
            if self.rec==True:
                painter.drawRect(500,200,740,400)
            self.lb2.show()
            self.lb1.close()
            self.lb2.move(245,35)
            self.lb2.setFont(QFont("Marker Felt", 20, QFont.Black))
            self.lb2.adjustSize()

        pen = QPen(Qt.black, 6, Qt.SolidLine)
        painter.setPen(pen)

        if len(pos_xy) > 1:
            point_start = pos_xy[0]
            for pos_tmp in pos_xy:
                point_end = pos_tmp
                if point_end == (-1, -1,-1):
                    point_start = (-1, -1,-1)
                    continue
                if point_start == (-1, -1,-1):
                    point_start = point_end
                    continue
                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()
        self.update()
    def buttonClicked2(self):
        global STOPFLAG
        global DIS
        global STOPGESTURE
        STOPGESTURE=False
        self.rec=True
        self.image.close()
        self.showname.close()
        self.similarity.close()
        self.log.close()
        self.signature=[]
        self.distance=[]
        self.minindex=[]
        self.minin=[]
        STOPFLAG=False
        DIS=1
        SpeedClear()
        self.qbtn2.close()
        self.qbtn.show()
        self.update()
    def buttonClicked(self):
        global dataset
        global Person
        global Text
        global pos_xy
        global STOPFLAG
        global DIS
        global speeda,speedb,speedc
        global Calstep
        global STOPGESTURE
        STOPGESTURE=True
        if pos_xy==[(-1,-1,-1)] or len(pos_xy)==2:
            QMessageBox.information(self,"Warning", "Please sign your name!")
            return
        while (-1, -1, -1) in pos_xy:
            pos_xy.remove((-1, -1, -1))
        pos_xy=pos_xy[::2]
        # pos_xy=pos_xy[::2]
        Csa, Csb, Csr = Direction(pos_xy)
        Sa, Sb, Sc = Slope(pos_xy)
        # A1,A2,A3=Aspect(pos_xy)
        acc1,acc2,acc3=Acceleration(speeda,speedb,speedc)
        self.signature = zip(Csa, Csb, Csr, Sa, Sb, Sc,speeda,speedb,speedc,acc1,acc2,acc3)
        self.cal.show()
        self.pbar.show()
        self.timer.start(self.maxnum+2, self)
        self.update()

        db=self.Connectmysql()
        self.Selectmysql(db)
        name=self.Choosename(db)
        Threshold=self.getThreshold(db,name)
        self.timer.stop()
        Calstep=0
        self.Closemysql(db)
        pos_xy = []
        flag2=False
        for flag in self.index2:
            mindis=self.distance[self.minindex[flag]]
            mean=Threshold-6500
            if mindis<Threshold:
                simi=100*(mean-abs(mean-mindis))/mean
                simi=0.15*simi+85
                self.picture = QtGui.QPixmap('D:/ISSD_image/%s.jpeg'%name)
                self.image.setPixmap(self.picture)
                self.image.show()
                self.showname.setText(name)
                self.showname.adjustSize()
                self.showname.show()
                self.log.show()
                self.similarity.setText("Similarity:%.2f%%"%simi)
                self.similarity.adjustSize()
                self.similarity.show()
                flag2=True
                break
        if flag2!=True:
            self.showname.setText('     Deny Login')
            self.showname.adjustSize()
            self.showname.show()
        self.pbar.close()
        self.cal.close()
        print "minidex,%s"%self.minindex
        print "distacne,%s" % self.distance
        print "mindistance,%s,%s,%s"%(self.distance[self.minindex[0]],self.distance[self.minindex[1]],self.distance[self.minindex[2]])
        self.qbtn.close()
        self.qbtn2.show()
        self.rec=False
        self.update()



    def buttoncenter(self,button):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        button.move(qr.center())
    def end(self):
        global pos_xy
        global STOPFLAG
        global DIS
        global STOPGESTURE
        self.controller.remove_listener(self.listener)
        STOPGESTURE=False
        self.rec=True
        self.image.close()
        self.showname.close()
        self.similarity.close()
        self.log.close()
        self.signature=[]
        self.distance=[]
        self.minindex=[]
        self.minin=[]
        pos_xy=[]
        SpeedClear()
        STOPFLAG=False
        DIS=1
        self.qbtn2.close()
        self.qbtn.show()
        self.update()
        self.close()


    def display(self):
        self.show()

    def getNameid(self,db,name):
        nameid=[]
        cursor=db.cursor()
        sql = "select id from signature where Person='%s'"%name
        cursor.execute(sql)
        result = cursor.fetchall()
        for i in range(len(result)):
            nameid.append(result[i][0])
        db.commit()
        cursor.close()
        print nameid
        return nameid


    def getThreshold(self,db,name):
        global Calstep
        global tdis
        cursor = db.cursor()
        distance=[]
        nameid=self.getNameid(db,name)
        if len(nameid)!=1:
            for i in range(len(nameid)):
                tablenum = self.gettablenum(db,nameid[i])
                for aa in range(tablenum):
                    sql = "select Dataset from %s where id=%d" % (tablename[aa],nameid[i])
                    cursor.execute(sql)
                    db.commit()
                    if aa == 0:
                        result = literal_eval(cursor.fetchone()[0])
                    else:
                        result = result + literal_eval(cursor.fetchone()[0])
                if i >= 1:
                    distance.append(fastdtw.fastdtw(result,tempresult)[0])
                tempresult = result
                Calstep=Calstep+1
                self.pbar.setValue(Calstep)
        else:
            Threashold=4000
        cursor.close()
        print "distance in sample:%s"%distance
        meandis=numpy.mean(distance)
        # std=numpy.var(distance)**0.5
        # Threashold=meandis+tdis[len(distance)-2]*std/(len(distance)**0.5)
        Threashold=meandis+3000
        print "threshold:%s"%Threashold
        return Threashold


    def Choosename(self,db):
        global KNN_K
        self.index2=[]
        name=[]
        for i in range(KNN_K):
            name.append(self.Getname(db,self.minindex[i]))
        namecount=Counter(name)
        namemost=namecount.most_common(1)[0][0]
        self.namemosttime=namecount[namemost]
        for item in enumerate(name):
            if item[1]==namemost:
                self.index2.append(item[0])
        print name, namemost, namecount,self.index2,self.minindex
        if self.namemosttime!=1:
            return namemost
        else:
            return name[0]


    def Getname(self,db,minindex):
        cursor=db.cursor()
        sql="select Person from signature where id='%d'"%minindex
        cursor.execute(sql)
        db.commit()
        name=cursor.fetchone()[0]
        cursor.close()
        return name
    def Connectmysql(self):
        print 'connecting to database......'
        conn = MySQLdb.connect(
            host='localhost',
            port=3306,
            user='root',

            passwd='qweruiop',
            db='test',
            charset='utf8',
        )
        print 'connected'
        return conn

    def Closemysql(self,db):
        db.close()

    def getrownum(self,db):
        cursor=db.cursor()
        sql="select * from signature"
        cursor.execute(sql)
        db.commit()
        crossnum=cursor.rowcount
        cursor.close()
        return crossnum
    def gettablenum(self,db,i):
        cursor=db.cursor()
        sql="select tablenum from signature1 where id='%d'"%i
        cursor.execute(sql)
        db.commit()
        tablenum = cursor.fetchone()[0]
        cursor.close()
        return tablenum

    def Selectmysql(self,db):
        global tablename
        global KNN_K
        global Calstep
        cursor = db.cursor()
        rownum=self.getrownum(db)
        print 'start selecting.......'
        for i in range(rownum):
            tablenum=self.gettablenum(db,i)
            for aa in range(tablenum):
                sql = "select Dataset from %s where id=%d" % (tablename[aa],i)
                cursor.execute(sql)
                db.commit()
                if aa == 0:

                    result = literal_eval(cursor.fetchone()[0])
                else:
                    result = result + literal_eval(cursor.fetchone()[0])

            self.distance.append(fastdtw.fastdtw(result,self.signature)[0])
            Calstep=Calstep+1
            self.pbar.setValue(Calstep)
        self.minin=heapq.nsmallest(KNN_K,self.distance)

        for mm in range(KNN_K):
            self.minindex.append(self.distance.index(self.minin[mm]))


        cursor.close()
        print 'select successfully!'





class Input(QWidget):
    def __init__(self):
        super(Input,self).__init__()

        self.initUI()
    def initUI(self):

        self.okButton = QPushButton("OK")
        self.cancelButton = QPushButton("Cancel")


        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.okButton.clicked.connect(self.okbutton)
        self.okButton.setShortcut(Qt.Key_Return)
        self.cancelButton.clicked.connect(self.cancelbutton)

        self.setLayout(vbox)

        self.resize(300,300)
        self.center()
        self.setWindowTitle('Input your name')

        self.le=QLineEdit(self)
        self.le.move(100,100)

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(92,150,150, 15)
        self.pbar.setMaximum(20)
        self.pbar.close()

        self.label=QLabel('Taking pictures',self)
        self.label.setGeometry(105,175,150, 20)
        self.label.adjustSize()
        self.label.close()

        self.timer = QBasicTimer()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def buttoncenter(self,button):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        button.move(qr.topLeft())

    def okbutton(self):
        global Text
        Text=self.le.text()
        self.pbar.show()
        self.timer.start(20, self)
        self.label.show()
        self.cap = cv2.VideoCapture(0)
        for ttt in range(20):
            self.ret, self.frame = self.cap.read()
            self.pbar.setValue(ttt)
            self.update()
        cv2.imwrite("D:/ISSD_image/%s.jpeg"%Text, self.frame)
        self.pbar.close()
        self.timer.stop()
        self.label.close()

        self.le.setText('')
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()


    def cancelbutton(self):
        self.le.setText('')
        self.close()

    def display(self):
        self.show()



class SampleListener(Leap.Listener):
    def on_init(self, controller):
        print "Initialized"
        #self.window()
        self.pos_tmp=(960,540)
        self.pos_test=(-1,-1,-1)
        self.temp2=(-1,-1,-1)
        self.count=0
        self.temppos=[]
    def on_connect(self, controller):
        print "Connected"

    def on_disconnect(self, controller):
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"

    def on_frame(self, controller):
        global pos_xy
        global STOPFLAG
        global DIS
        global Hand_type
        global STOPGESTURE
        frame = controller.frame()
        gesture=Leap.Gesture
        hand=frame.hands.frontmost
        controller.enable_gesture(gesture.TYPE_SWIPE,True)
        pointable = frame.pointables.frontmost
        speed = pointable.tip_velocity
        a=speed[0]
        b=speed[1]
        c=speed[2]
        abc=(a,b,c)
        stabilizedPosition = pointable.stabilized_tip_position
        self.pos_tmp = (875+stabilizedPosition[0], 605-stabilizedPosition[1],stabilizedPosition[2])
        if abs(self.pos_tmp[0]-self.temp2[0])<=4 and abs(self.pos_tmp[1]-self.temp2[1])<=4 and abs(self.pos_tmp[2]-self.temp2[2])<=4:
            if frame.fingers.is_empty:
                speed=(-510,-510,-510)
            if 0<=abs(speed[0])<100 and 0<=abs(speed[1])<100 and 0<=abs(speed[2])<100:
                self.count = self.count + 1
                if self.count >= 50:
                    DIS = -1
                    self.count = 0
                if len(pos_xy) >= 200 and DIS==-1 and self.count >= 49:
                    STOPFLAG = True
                    self.count = 0

            else:
                self.count=0

        self.temp2=self.pos_tmp

        if (pointable.touch_distance <=DIS or pointable.touch_zone == Leap.Pointable.ZONE_NONE):
            if STOPFLAG == False:
                if len(pos_xy)<40:
                    pos_xy = []
                    SpeedClear()
                else:
                    pass
            else:
                pass


        if self.pos_tmp!=(875,605,0):
            if STOPFLAG==False:
                pos_xy.append(self.pos_tmp)
                pos_xy.append(self.pos_tmp)
                SpeedApend(a,b,c)

        if frame.fingers.is_empty and frame.tools.is_empty:
            if STOPFLAG==False:
                self.pos_test = (-1, -1, -1)
                pos_xy.append(self.pos_test)


        for gesture in frame.gestures():
            if Hand_type == True:
                type=hand.is_left
            else:
                type=hand.is_right
            if gesture.type is Leap.Gesture.TYPE_SWIPE and type and STOPGESTURE==False:
                    pos_xy = []
                    SpeedClear()
                    DIS = 1
                    STOPFLAG=False



def SpeedApend(a,b,c):
    global speeda
    global speedb
    global speedc
    speeda.append(a/10)
    speedb.append(b/10)
    speedc.append(c/10)

def SpeedClear():
    global speeda
    global speedb
    global speedc
    speeda=[]
    speedb=[]
    speedc=[]

def Aspect(list):
    A1=[]
    A2=[]
    A3=[]
    for t in range(1, len(list) - 1):
        dx = list[t - 1][0] - list[t + 1][0]
        dy = list[t - 1][1] - list[t + 1][1]
        dz = list[t - 1][2] - list[t + 1][2]
        a1=2*dy/(dx+dy)-1
        a2=2*dz/(dy+dz)-1
        a3=2*dz/(dz+dx)-1
        A1.append(a1)
        A2.append(a2)
        A3.append(a3)
    return A1,A2,A3

def Acceleration(a,b,c):
    acc1=[]
    acc2=[]
    acc3=[]
    for t in range(3, len(a) - 3):
        acc1.append((a[t+3]-a[t-3])/15)
        acc2.append((b[t+3]-b[t-3])/15)
        acc3.append((c[t+3]-c[t-3])/15)
    return acc1,acc2,acc3


def Direction(list):
    csa = []
    csb = []
    csr = []
    for t in range(1, len(list) - 1):
        dx = list[t - 1][0] - list[t + 1][0]
        dy = list[t - 1][1] - list[t + 1][1]
        dz = list[t - 1][2] - list[t + 1][2]
        ds = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        csa.append(5*float(dx) /float(ds))
        csb.append(5*float(dy) / float(ds))
        csr.append(5*float(dz) / float(ds))
    return csa, csb, csr

def Slope(list):
    a=[]
    b=[]
    c=[]
    for t in range(3, len(list) - 3):
        a1=list[t+3][0]-list[t-3][0]
        b1=list[t+3][0]-list[t-3][1]
        c1=list[t+3][2]-list[t-3][2]
        s=(a1**2+b1**2+c1**2)**0.5
        a.append(1000*float(a1)/float(s))
        b.append(1000*float(b1)/float(s))
        c.append(1000*float(c1)/float(s))
    return a, b, c

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex=MainWindow()
    inp=Input()
    test=TestWindow()
    draw=DrawWindow()

    ex.trainbutton.clicked.connect(inp.display)
    ex.testbutton.clicked.connect(test.display)
    inp.okButton.clicked.connect(draw.display)


    sys.exit(app.exec_())
