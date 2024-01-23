import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, \
    QTableWidget, QTableWidgetItem, QHeaderView, QPlainTextEdit

from engine import TextImprovementEngine


class GUIApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Text Improvement Engine')
        self.setGeometry(100, 100, 800, 800)

        layout = QVBoxLayout()

        # input text box
        self.input_text = QPlainTextEdit(self)
        self.input_text.setPlaceholderText("Enter text here...")
        layout.addWidget(self.input_text)

        # analyse button
        self.analyse_button = QPushButton('Analyse', self)
        self.analyse_button.clicked.connect(self.analyse_text)
        layout.addWidget(self.analyse_button)

        # table for results
        self.results_table = QTableWidget(self)
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(
            ["Original Phrase", "Suggested Phrase", "Similarity Score"])

        # set the resize mode for the first two columns to stretch
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        # set a fixed width for the "Similarity Score" column
        self.results_table.setColumnWidth(2, 100)

        # add the table to the layout
        layout.addWidget(self.results_table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def analyse_text(self):
        # if there's no text, do nothing
        text = self.input_text.toPlainText()
        if not text:
            return

        # clear the table
        self.results_table.setRowCount(0)

        # process the text
        engine = TextImprovementEngine()
        results = engine.analyse_text(text)

        # populate the table
        for result in results.values():
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)
            self.results_table.setItem(row_position, 0, QTableWidgetItem(result.original_phrase))
            self.results_table.setItem(row_position, 1, QTableWidgetItem(result.suggested_phrase))
            self.results_table.setItem(row_position, 2,
                                       QTableWidgetItem(f"{result.similarity_score:.3f}"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUIApp()
    ex.show()
    sys.exit(app.exec_())
