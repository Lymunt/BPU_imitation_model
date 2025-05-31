import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import csv


class PredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Branch Predictor Interface")
        self.setGeometry(100, 100, 1000, 600)

        self.layout = QVBoxLayout()

        self.load_button = QPushButton("Загрузить trace-файл")
        self.load_button.clicked.connect(self.load_trace_file)
        self.layout.addWidget(self.load_button)

        self.run_button = QPushButton("Запустить предсказание")
        self.run_button.clicked.connect(self.run_prediction)
        self.run_button.setEnabled(False)
        self.layout.addWidget(self.run_button)

        self.weights_button = QPushButton("Показать таблицу персептронов")
        self.weights_button.clicked.connect(self.show_weights_table)
        self.weights_button.setEnabled(False)
        self.layout.addWidget(self.weights_button)

        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)

        self.final_accuracy_label = QLabel("")
        self.final_accuracy_label.setStyleSheet("font-weight: bold; padding: 4px;")
        self.layout.addWidget(self.final_accuracy_label)

        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.canvas = FigureCanvas(plt.Figure(figsize=(5, 2)))
        self.layout.addWidget(self.canvas)

        self.reset_button = QPushButton("Сбросить модель")
        self.reset_button.clicked.connect(self.reset_predictor)
        self.reset_button.setEnabled(False)
        self.layout.addWidget(self.reset_button)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        from predict_module import BranchPredictor
        self.predictor = BranchPredictor()
        from predict_module import LoopPredictor
        self.loop_predictor = LoopPredictor()
        self.trace_path = None

    def load_trace_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выбери trace.txt", "", "Text Files (*.txt)")
        if file_name:
            self.trace_path = file_name
            self.result_label.setText(f"Загружен файл: {file_name}")
            self.run_button.setEnabled(True)
            self.reset_button.setEnabled(True)

    def run_prediction(self):
        self.canvas.figure.clf()
        
        from predict_module import run_predictor
        run_predictor(trace_file=self.trace_path, output_csv="gui_results.csv", predictor=self.predictor, loop_predictor=self.loop_predictor)

        with open("gui_results.csv", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)

        current_rows = self.table.rowCount()
        self.table.setColumnCount(len(headers))
        self.table.setRowCount(current_rows + len(rows))
        self.table.setHorizontalHeaderLabels(headers)

        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                if j == 4 and float(val) < 1.0:
                    item.setBackground(Qt.GlobalColor.yellow)
                self.table.setItem(current_rows + i, j, item)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        ax = self.canvas.figure.subplots()
        ax.clear()
        accuracies = [float(row[4]) for row in rows]
        ax.plot(range(len(accuracies)), accuracies, label='Accuracy')
        ax.set_title("Точность предсказаний")
        ax.set_xlabel("Шаг")
        ax.set_ylabel("Точность")
        ax.legend()
        self.canvas.draw()

        final_accuracy = float(rows[-1][4]) if rows else 0.0
        self.final_accuracy_label.setText(f"Итоговая точность: {final_accuracy:.2%}")

        self.weights_button.setEnabled(True)

    def reset_predictor(self):
        from predict_module import BranchPredictor
        self.predictor = BranchPredictor()
        from predict_module import LoopPredictor 
        self.loop_predictor = LoopPredictor()
        self.final_accuracy_label.setText("")
        self.result_label.setText("Модель сброшена")
        self.weights_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.trace_path = None

        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        self.canvas.figure.clf()
        self.canvas.draw()

    def show_weights_table(self):
        weights = self.predictor.get_perceptron_weights()

        weight_table = QTableWidget()
        weight_table.setWindowTitle("Таблица весов персептронов")
        weight_table.setColumnCount(len(weights[0]))
        weight_table.setRowCount(len(weights))
        weight_table.resize(1000, 600)

        headers = ["Bias"] + [f"w{j}" for j in range(1, len(weights[0]))]
        weight_table.setHorizontalHeaderLabels(headers)

        row_labels = [f"P{i}" for i in range(len(weights))]
        weight_table.setVerticalHeaderLabels(row_labels)

        for i, row in enumerate(weights):
            for j, val in enumerate(row):
                weight_table.setItem(i, j, QTableWidgetItem(str(val)))

        weight_table.show()
        self.weights_window = weight_table 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictorApp()
    window.show()
    sys.exit(app.exec())
