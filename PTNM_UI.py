# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Recommend_sys_ver2_1 import *

# widget UI setting
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        """
        초기설정
        """
        self.setWindowTitle("PNTM System")
        self.setWindowIcon(QIcon('music_icon.png'))
        self.setGeometry(300, 300, 300, 300)
        self.setStyleSheet("background-color : #ff9933")
        self.stack = QStackedWidget(self)  # stackWidget 정의

        """
        필요한 위젯들 만들기
        """
        self.lineEdit1 = QLineEdit(self)  # 사용자가 입력한 노래 5곡 입력받음
        self.lineEdit2 = QLineEdit(self)
        self.lineEdit3 = QLineEdit(self)
        self.lineEdit4 = QLineEdit(self)
        self.lineEdit5 = QLineEdit(self)

        self.input1 = QLabel(self)
        self.input2 = QLabel(self)
        self.input3 = QLabel(self)
        self.input4 = QLabel(self)
        self.input5 = QLabel(self)

        self.output1 = QLabel(self)     # 추천한 노래 5곡
        self.output2 = QLabel(self)
        self.output3 = QLabel(self)
        self.output4 = QLabel(self)
        self.output5 = QLabel(self)

        self.output_more1 = QLabel(self)        # 추천한 노래 5곡
        self.output_more2 = QLabel(self)
        self.output_more3 = QLabel(self)
        self.output_more4 = QLabel(self)
        self.output_more5 = QLabel(self)

        self.lineEdit = []
        self.input = []
        self.output = []
        self.output_more = []
        self.combos_01 = []
        self.combos_02 = []
        for i in range(0, 5):
            self.lineEdit.append(QLineEdit(self))   # 사용자가 입력한 노래 5곡 입력받음
            self.lineEdit[i].setFont(QFont('PFStardust', 15))
            self.input.append(QLabel(self))
            self.output.append(QLabel(self))        # 추천한 노래 5곡
            self.output_more.append(QLabel(self))   # 추천한 노래 5곡
            self.combos_01.append(QComboBox())  # 노래 5곡에 대한 각각의 별점
            self.combos_01[i].addItems(['★', '★★', '★★★', '★★★★', '★★★★★'])
            self.combos_02.append(QComboBox())
            self.combos_02[i].addItems(['★', '★★', '★★★', '★★★★', '★★★★★'])

        self.stack0 = QWidget()
        self.stack1 = QWidget()
        self.stack2 = QWidget()
        self.stack3 = QWidget()
        self.stack4 = QWidget()

        self.mainwindow()  # 메인 화면 # stack0
        self.inputmusics()  # stack1 # stack2
        self.outputmusics()  # stack3 # stack4

        self.stack.addWidget(self.stack0)
        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)
        self.stack.addWidget(self.stack3)
        self.stack.addWidget(self.stack4)

        # 전체 레이아웃
        entirelayout = QVBoxLayout()
        entirelayout.addWidget(self.stack)

        self.setLayout(entirelayout)
        self.display(0)

    def mainwindow(self):  # 메인 화면 #stack0

        agumon = QLabel(self)
        pixmap = QPixmap('agumon_pixel_withcircle.png')
        width_label = 130
        height_label = 130
        agumon.resize(width_label, height_label)
        agumon.setPixmap(pixmap.scaled(agumon.size(), Qt.IgnoreAspectRatio))  # 아구몬 사진
        agumon.setAlignment(Qt.AlignCenter)

        label1 = QLabel("Welcome to PTNM")
        label1.setAlignment(Qt.AlignCenter)
        label1.setFont(QFont("Press Start 2P",20))
        label2 = QLabel("I'll recommend a song that suits your taste.")
        label2.setAlignment(Qt.AlignCenter)
        label2.setFont(QFont("Press Start 2P", 10))

        pushButton0_1 = QPushButton("START")  # 노래 입력 창 연결
        pushButton0_2 = QPushButton("EXIT")
        pushButton0_1.setFont(QFont("Press Start 2P", 10))
        pushButton0_2.setFont(QFont("Press Start 2P", 10))

        buttonlayout = QHBoxLayout()
        buttonlayout.addWidget(pushButton0_1)
        buttonlayout.addWidget(pushButton0_2)

        layout = QVBoxLayout()
        layout.addWidget(agumon)
        layout.addWidget(label1)
        layout.addWidget(label2)
        layout.addLayout(buttonlayout)

        self.stack0.setLayout(layout)
        pushButton0_1.clicked.connect(lambda: self.display(1))  # stack1으로 넘어감
        pushButton0_2.clicked.connect(QCoreApplication.instance().quit)  # 종료


    def inputmusics(self):  # 노래 받고 제목 출력해서 사용자에게 확인받기 #stack1 #stack2
        """
        # input musics #stack1
        """
        # 노래입력부분


        label1_0 = QLabel("Please enter your favorite songs!")
        label1_0.setFont(QFont("Press Start 2P", 13))

        label1 = []
        for i in range(0, 5):
            name = 'music ' + str(i+1) + ': '
            label1.append(QLabel(name))
            label1[i].setFont(QFont("Press Start 2P", 10))

        Inputlayout = QGridLayout()
        Inputlayout.addWidget(label1_0, 0, 1)
        for i in range(0, 5):
            Inputlayout.addWidget(label1[i], i+1, 0)
            Inputlayout.addWidget(self.lineEdit[i], i+1, 1)

        # 버튼 부분
        pushButton1_1 = QPushButton("SUBMIT")
        pushButton1_2 = QPushButton("HOME")
        pushButton1_1.setFont(QFont("Press Start 2P", 10))
        pushButton1_2.setFont(QFont("Press Start 2P", 10))

        buttonlayout1 = QHBoxLayout()
        buttonlayout1.addWidget(pushButton1_1)
        buttonlayout1.addWidget(pushButton1_2)

        layout1 = QVBoxLayout()
        layout1.addLayout(Inputlayout)
        layout1.addLayout(buttonlayout1)
        self.stack1.setLayout(layout1)
        pushButton1_1.clicked.connect(lambda: self.lineedited(2))  # stack2으로 넘어감
        pushButton1_2.clicked.connect(lambda: self.display(0))  # stack0으로 넘어감

        """
        # stack2
        """
        print('stack 02')
        # 노래 확인 부분
        label2_0 = QLabel("Check Your Submission")
        label2_0.setFont(QFont("Press Start 2P", 13))

        Confirmlayout = QVBoxLayout()
        Confirmlayout.addWidget(label2_0)
        for i in range(0, 5):
            Confirmlayout.addWidget(self.input[i])

        # 버튼 부분
        pushButton2_1 = QPushButton("RECOMMEND")
        pushButton2_2 = QPushButton("HOME")

        pushButton2_1.setFont(QFont("Press Start 2P", 10))
        pushButton2_2.setFont(QFont("Press Start 2P", 10))

        buttonlayout2 = QHBoxLayout()
        buttonlayout2.addWidget(pushButton2_1)
        buttonlayout2.addWidget(pushButton2_2)

        layout2 = QVBoxLayout()
        layout2.addLayout(Confirmlayout)
        layout2.addLayout(buttonlayout2)

        self.stack2.setLayout(layout2)
        pushButton2_1.clicked.connect(lambda: self.display(3))  # stack3으로 넘어감
        pushButton2_2.clicked.connect(lambda: self.display(0))  # 메인화면으로 넘어감

    def outputmusics(self):
        """
        # stack3
        """
        # 노래 출력 부분

        groupBox1 = QGroupBox("     PTNM recommends 5 songs to you!")
        groupBox1.setFont(QFont("Press Start 2P", 10))
        musiclayout = QVBoxLayout()
        for i in range(0, 5):
            musiclayout.addWidget(self.output[i])  # 추천해줄 노래 5곡 문자열로 반환

        groupBox1.setLayout(musiclayout)

        leftlayout = QVBoxLayout()
        leftlayout.addWidget(groupBox1)

        groupBox2 = QGroupBox("Please Enter Your Evaluation!")
        groupBox2.setFont(QFont("Press Start 2P", 10))

        gradelayout = QVBoxLayout()
        for i in range(0, 5):
            gradelayout.addWidget(self.combos_01[i])
        groupBox2.setLayout(gradelayout)

        rightlayout = QVBoxLayout()
        rightlayout.addWidget(groupBox2)

        # musiclayout 과 gradelayout 합치기
        checklayout = QHBoxLayout()
        checklayout.addWidget(groupBox1)
        checklayout.addWidget(groupBox2)

        label1 = QLabel("PTNM made by team 'Find a Key !!'\n from Dongguk Univ.")
        label1.setFont(QFont("Press Start 2P", 5))

        pushButton3_1 = QPushButton("SUBMIT")
        pushButton3_2 = QPushButton("HOME")

        pushButton3_1.setFont(QFont("Press Start 2P", 10))
        pushButton3_2.setFont(QFont("Press Start 2P", 10))

        buttonlayout = QHBoxLayout()
        buttonlayout.addWidget(label1)
        buttonlayout.addWidget(pushButton3_1)
        buttonlayout.addWidget(pushButton3_2)

        layout3 = QVBoxLayout()
        layout3.addLayout(checklayout)
        layout3.addLayout(buttonlayout)

        self.stack3.setLayout(layout3)
        pushButton3_1.clicked.connect(self.recommend)       # stack4로 넘어감
        pushButton3_2.clicked.connect(lambda: self.display(0))

        """
        # stack4
        """
        # 노래 출력
        # output another recommended musics
        groupBox1_2nd = QGroupBox("     PTNM recommends 5 songs again !!")
        groupBox1_2nd.setFont(QFont("Press Start 2P", 10))
        musiclayout_2nd = QVBoxLayout()
        for i in range(0, 5):
            musiclayout_2nd.addWidget(self.output_more[i])
        groupBox1_2nd.setLayout(musiclayout_2nd)

        leftlayout4 = QVBoxLayout()
        leftlayout4.addWidget(groupBox1_2nd)

        # 별점 콤보박스

        groupBox2_2nd = QGroupBox("Please Enter Your Evaluation!")
        groupBox2_2nd.setFont(QFont("Press Start 2P", 10))
        gradelayout_2nd = QVBoxLayout()
        for i in range(0, 5):
            gradelayout_2nd.addWidget(self.combos_02[i])
        groupBox2.setLayout(gradelayout)
        groupBox2_2nd.setLayout(gradelayout_2nd)

        rightlayout5 = QVBoxLayout()
        rightlayout5.addWidget(groupBox2_2nd)

        # musiclayout_2nd 하고 gradelayout_2nd 합치기
        toplayout = QHBoxLayout()
        toplayout.addWidget(groupBox1_2nd)
        toplayout.addWidget(groupBox2_2nd)

        pushButton4_0 = QPushButton("SUBMIT")
        pushButton4_1 = QPushButton("EXIT")
        pushButton4_2 = QPushButton("HOME")

        pushButton4_0.setFont(QFont("Press Start 2P", 10))
        pushButton4_1.setFont(QFont("Press Start 2P", 10))
        pushButton4_2.setFont(QFont("Press Start 2P", 10))

        pushButton4_0.clicked.connect(self.re_recommend)
        pushButton4_1.clicked.connect(QCoreApplication.instance().quit)
        pushButton4_2.clicked.connect(lambda: self.display(0))

        buttonlayout4 = QHBoxLayout()
        buttonlayout4.addWidget(pushButton4_0)
        buttonlayout4.addWidget(pushButton4_1)
        buttonlayout4.addWidget(pushButton4_2)

        layout4 = QVBoxLayout()
        layout4.addLayout(toplayout)
        layout4.addLayout(buttonlayout4)
        self.stack4.setLayout(layout4)

    @pyqtSlot(int)
    def display(self, i):
        self.stack.setCurrentIndex(i)

    def changeFont(self):
        font = QFont('fontYouandiModernTR')
        self.setFont(font)

    def lineedited(self, i):
        input_list = []
        for j in range(0, 5):
            self.input[j].setText(self.lineEdit[j].text())
            self.input[j].setFont(QFont('PFStardust', 15))
            if self.lineEdit[j].text() != '':
                input_list.append(self.lineEdit[j].text())

        print('input :: ', input_list)

        self.recom = PTNM_Recommend(input_list)
        self.recom.start_recom()
        print('recom_song_list :: ', self.recom.recom_song_list)
        for j in range(0, 5):
            self.output[j].setText(self.recom.recom_song_list[j])
            self.output[j].setFont(QFont("PFStardust", 15))

        self.stack.setCurrentIndex(i)

    def comboboxIndexChanged(self):
        self.grade1 = self.combo1.currentIndex() -10
        self.grade2 = self.combo2.currentIndex() -10
        self.grade3 = self.combo3.currentIndex() -10
        self.grade4 = self.combo4.currentIndex() -10
        self.grade5 = self.combo5.currentIndex() -10

        print(self.grade1, self.grade2, self.grade3, self.grade4, self.grade5)

    def recommend(self):
        feedback_estim = []
        for i in range(0, 5):
            feedback_estim.append(self.combos_01[i].currentIndex() * 5 - 10)
        print('feedback_estim :: ', feedback_estim)

        self.recom.feedback_input(feedback_estim)
        self.recom.recom_steps()

        for i in range(0, 5):
            self.output_more[i].setText(self.recom.recom_song_list[i])  # 재추천해줄 노래 5곡 문자열로 반환
            self.output_more[i].setFont(QFont('PFStardust', 15))

        self.display(4)

    def re_comboboxIndexChanged(self):
        self.grade_2nd_1 = self.combo_2nd_1.currentIndex() -10
        self.grade_2nd_2 = self.combo_2nd_2.currentIndex() -10
        self.grade_2nd_3 = self.combo_2nd_3.currentIndex() -10
        self.grade_2nd_4 = self.combo_2nd_4.currentIndex() -10
        self.grade_2nd_5 = self.combo_2nd_5.currentIndex() -10

        print(self.grade_2nd_1, self.grade_2nd_2, self.grade_2nd_3, self.grade_2nd_4, self.grade_2nd_5)


    def re_recommend(self):
        feedback_estim = []
        for i in range(0, 5):
            feedback_estim.append(self.combos_02[i].currentIndex() * 5 - 10)
        self.recom.feedback_input(feedback_estim)
        self.recom.recom_steps()

        for i in range(0, 5):
            self.output_more[i].setText(self.recom.recom_song_list[i])  # 재추천해줄 노래 5곡 문자열로 반환
            self.output_more[i].setFont(QFont('PFStardust', 15))

        self.display(4)

# main loop
class Mainloop(QObject):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.gui = MyWindow()
        self.gui.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("YouandiModern.ttf"))
    mainloop = Mainloop(app)
    sys.exit(app.exec_())