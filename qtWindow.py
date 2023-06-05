import sys
import os
import time
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QRect, QEvent, QObject
from PyQt5.QtGui import QScreen, QGuiApplication


from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from jdChart import JdChart

data_folder = os.path.join(os.getcwd(), 'UI')
ui_path = os.path.join(data_folder, 'qtWindow.ui')

form_class = uic.loadUiType(ui_path)[0]

class JdWindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.chart : JdChart = None

        screen = QGuiApplication.primaryScreen()
        screen_size = screen.availableSize()
        window_width = screen_size.width() * 0.9  # 전체 화면 폭의 80%
        window_height = screen_size.height() * 0.9  # 전체 화면 높이의 80%
        self.resize(window_width, window_height)

        self.label : QLabel
        self.setWindowTitle("Jidon stock helper")

        self.lineEdit_search : QLineEdit
        self.lineEdit_search.returnPressed.connect(self.on_lienEdit_search_returnPressed)

        self.checkbox_ma10 : QCheckBox
        self.checkbox_ma20 : QCheckBox

        self.checkbox_ma10.stateChanged.connect(self.on_ma_checkbox_changed)
        self.checkbox_ma20.stateChanged.connect(self.on_ma_checkbox_changed)


        self.setFocusPolicy(Qt.StrongFocus)

    def set_chart_class(self, inChartClass : JdChart):
        self.chart = inChartClass
        fig = self.chart.draw_stock_chart()
        self.set_canvas(fig)


    def refresh_canvas(self):
        #self.canvas.updateGeometry()
        self.canvas.draw()


    def set_canvas(self, inFig : Figure):
        # FigureCanvas 생성 및 설정
        self.canvas = FigureCanvasQTAgg(inFig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setParent(self.label)
        print('canvas size hint: ', self.canvas.sizeHint())
        print('label size: ', self.label.size().width(), self.label.size().height())
        self.canvas.setFixedSize(self.label.size())
        self.refresh_canvas()


            
    def on_lienEdit_search_returnPressed(self):
        targetTicker = self.lineEdit_search.text()
        print(targetTicker)

        res = self.chart.move_to_ticker_stock(targetTicker)
        if res:
            self.chart.move_to_ticker_stock(targetTicker)
            self.chart.draw_stock_chart()
            self.refresh_canvas()

    def on_ma_checkbox_changed(self):
        bMa10Checked = self.checkbox_ma10.isChecked()
        bMa20Checked = self.checkbox_ma20.isChecked()

        self.chart.set_ma_visibility(bMa10Checked, bMa20Checked)
        self.chart.draw_stock_chart()
        self.refresh_canvas()



    def keyPressEvent(self, event):
        start_time = time.time()

        key = event.key()
        if event.key() == Qt.Key_Escape:
            #print('Qt.Key_Escape')
            self.chart.on_close(event)
            self.close()
            sys.exit(1)
        elif event.key() == Qt.Key_Right:
            #print('Qt.Key_Right')
            self.chart.move_to_next_stock()
            self.chart.draw_stock_chart()
            self.refresh_canvas()
        elif event.key() == Qt.Key_Left:
            #print('Qt.Key_Left')
            self.chart.move_to_prev_stock()
            self.chart.draw_stock_chart()
            self.refresh_canvas()
        elif event.key() == Qt.Key_Return:
            #print('Qt.Key_Return')
            if not self.lineEdit_search.hasFocus():
                self.chart.mark_ticker()
        elapsedTime = time.time() - start_time
        #print('keyPressEvent handling time : ', elapsedTime)


        

# if __name__ == "__main__" :
#     #QApplication : 프로그램을 실행시켜주는 클래스
#     app = QApplication(sys.argv) 

#     #WindowClass의 인스턴스 생성
#     myWindow = JdWindowClass() 

#     #프로그램 화면을 보여주는 코드
#     myWindow.show()

#     #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
#     app.exec_()
