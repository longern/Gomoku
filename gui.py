import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.ui_board import Ui_Board
from board import Board
from gomoku_ai import ai_move
import random

class Ui(QtWidgets.QWidget):
	boardPos = (28, 28)
	gridSize = (33, 33)
	chessSize = (25, 25)
	starSize = (5, 5)

	def __init__(self, parent=None):
		super(Ui, self).__init__(parent)
		self.ui = Ui_Board()
		self.ui.setupUi(self)
		self.ui.choosePanel.hide()
		self.board = Board()
		self.ai = 0
		self.cursor_x = -1
		self.cursor_y = -1

	def paintEvent(self, QPaintEvent):
		painter = QtGui.QPainter(self)
		painter.setBrush(QtCore.Qt.black)
		for i in range(self.boardPos[0], self.boardPos[0] + self.gridSize[0] * 14 + 1, self.gridSize[0]):
			painter.drawLine(i, self.boardPos[1], i, self.boardPos[1] + self.gridSize[1] * 14)
		for i in range(self.boardPos[1], self.boardPos[1] + self.gridSize[1] * 14 + 1, self.gridSize[1]):
			painter.drawLine(self.boardPos[0], i, self.boardPos[0] + self.gridSize[0] * 14, i)
		for i in [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]:
			self.drawStar(painter, i)
		for i, _ in enumerate(self.board.history):
			self.drawChessPiece(painter, self.board.history[i], QtCore.Qt.white if i % 2 else QtCore.Qt.black)
		self.drawCursorPosition(painter, (self.cursor_x, self.cursor_y))
		if self.board.history:
			self.drawLastPlayPosition(painter, self.board.history[len(self.board.history) - 1])
		return super().paintEvent(QPaintEvent)

	def drawLastPlayPosition(self, painter, center):
		painter.setPen(QtCore.Qt.red)
		painter.translate(center[0] * self.gridSize[0] + self.boardPos[0], center[1] * self.gridSize[1] + self.boardPos[1])
		seg = self.gridSize[0] / 8
		thres = 1
		painter.drawLine(0, -seg, 0, -thres)
		painter.drawLine(0, seg, 0, thres)
		painter.drawLine(-seg, 0, -thres, 0)
		painter.drawLine(seg, 0, thres, 0)
		painter.resetTransform()

	def drawCursorPosition(self, painter, center):
		if not self.board.in_board(center):
			return
		painter.setPen(QtCore.Qt.black)
		painter.translate(center[0] * self.gridSize[0] + self.boardPos[0], center[1] * self.gridSize[1] + self.boardPos[1])
		seg = self.gridSize[0] / 4
		painter.drawLine(-seg * 2, -seg * 2, -seg, -seg * 2)
		painter.drawLine(-seg * 2, -seg * 2, -seg * 2, -seg)
		painter.drawLine(seg * 2, -seg * 2, seg, -seg * 2)
		painter.drawLine(seg * 2, -seg * 2, seg * 2, -seg)
		painter.drawLine(-seg * 2, seg * 2, -seg, seg * 2)
		painter.drawLine(-seg * 2, seg * 2, -seg * 2, seg)
		painter.drawLine(seg * 2, seg * 2, seg, seg * 2)
		painter.drawLine(seg * 2, seg * 2, seg * 2, seg)
		painter.resetTransform()

	def drawChessPiece(self, painter, center, color):
		painter.setBrush(color)
		painter.translate(center[0] * self.gridSize[0] + self.boardPos[0], center[1] * self.gridSize[1] + self.boardPos[1])
		rect = QtCore.QRectF(-self.chessSize[0] / 2, -self.chessSize[0] / 2, self.chessSize[0], self.chessSize[0]);
		painter.drawChord(rect, 0, 5760);
		painter.resetTransform()

	def drawStar(self, painter, center):
		painter.setBrush(QtCore.Qt.black)
		painter.translate(center[0] * self.gridSize[0] + self.boardPos[0], center[1] * self.gridSize[1] + self.boardPos[1])
		rect = QtCore.QRectF(-self.starSize[0] / 2, -self.starSize[0] / 2, self.starSize[0], self.starSize[0]);
		painter.drawChord(rect, 0, 5760);
		painter.resetTransform()

	def mouseMoveEvent(self, QMouseEvent):
		if self.board.winner:
			return super().mouseMoveEvent(QMouseEvent)
		self.cursor_x = int(round((QMouseEvent.x() - self.boardPos[0]) / self.gridSize[0]))
		self.cursor_y = int(round((QMouseEvent.y() - self.boardPos[1]) / self.gridSize[1]))
		self.update()
		return super().mouseMoveEvent(QMouseEvent)

	def mousePressEvent(self, QMouseEvent):
		if self.board.winner:
			return super().mousePressEvent(QMouseEvent)
		chess_x = int(round((QMouseEvent.x() - self.boardPos[0]) / self.gridSize[0]))
		chess_y = int(round((QMouseEvent.y() - self.boardPos[1]) / self.gridSize[1]))
		pos = (chess_x, chess_y)
		if self.board.in_board(pos) and self.board.at(pos) == 0:
			self.board.play(pos)
			self.afterPlay()
		self.update()
		return super().mousePressEvent(QMouseEvent)

	winning_message = {
		-1: "Draw!",
		1: "Black wins!",
		2: "White wins!"
	}

	def afterPlay(self):
		if self.board.winner:
			self.setWindowTitle(self.winning_message[self.board.winner])
			return
		if self.ai == self.board.current_player:
			self.board.play(ai_move(self.board))
			self.afterPlay()
		else:
			if self.ui.chkSwap2.checkState() == QtCore.Qt.Checked:
				if self.board.turn == 3:
					self.setWindowTitle("Swap 1")
				elif self.board.turn == 5:
					self.setWindowTitle("Swap 2")

	def on_btnHuman_clicked(self, checked=True):
		if checked:
			return
		self.board = Board()
		self.ai = 0
		self.setWindowTitle(QtCore.QCoreApplication.translate("Board", "Gomoku"))
		self.update()

	def on_btnAi_clicked(self, checked=True):
		if checked:
			return
		self.board = Board()
		self.ai = random.randint(1, 2)
		self.setWindowTitle(QtCore.QCoreApplication.translate("Board", "Gomoku"))
		self.afterPlay()
		self.update()
		self.update()

def gui_start():
	import sys
	app = QtWidgets.QApplication(sys.argv)
	app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
	window = Ui()
	window.show()
	sys.exit(app.exec_())
