from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QWidget,
    QMenuBar,
    QMenu,
    QTextEdit,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QStatusBar,
)
from PyQt5.QtCore import Qt
import sys
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from permanent import perm


class PermanentCalculatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Permanent Calculator")
        self.setGeometry(100, 100, 800, 600)
        self.matrix_data = None
        self.rows = None
        self.cols = None

        # menu bar
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)

        import_action = file_menu.addAction("Import File")
        import_action.triggered.connect(self.import_file)

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()

        self.status_label = QLabel("Import a file to calculate the permanent")
        self.main_layout.addWidget(self.status_label)

        h_layout = QHBoxLayout()

        matrix_group = QGroupBox("Matrix Data")
        matrix_layout = QVBoxLayout()
        self.matrix_display = QTextEdit()
        self.matrix_display.setReadOnly(True)
        matrix_layout.addWidget(self.matrix_display)
        matrix_group.setLayout(matrix_layout)
        h_layout.addWidget(matrix_group)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        results_layout.addWidget(self.results_display)
        results_group.setLayout(results_layout)
        h_layout.addWidget(results_group)

        self.main_layout.addLayout(h_layout)

        buttons_layout = QHBoxLayout()

        self.calculate_button = QPushButton("Calculate Permanent && Gradient")
        self.calculate_button.clicked.connect(self.calculate)
        self.calculate_button.setEnabled(False)
        buttons_layout.addWidget(self.calculate_button)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_data)
        buttons_layout.addWidget(clear_button)

        self.main_layout.addLayout(buttons_layout)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

    def import_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "NumPy Files (*.npy);;Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
            options=options,
        )

        if file_path:
            try:
                if file_path.endswith(".npy"):
                    data = np.load(file_path)
                    if not isinstance(data, np.ndarray):
                        raise TypeError(
                            "Loaded .npy file does not contain a NumPy array."
                        )
                    # Ensure it's 2D
                    if data.ndim != 2:
                        raise ValueError(
                            f"Expected a 2D array from .npy file, but got shape {data.shape}"
                        )
                elif file_path.endswith(".xlsx"):
                    data = pd.read_excel(file_path, header=None).to_numpy()
                elif file_path.endswith(".csv"):
                    data = pd.read_csv(file_path, header=None).to_numpy()
                else:
                    raise ValueError("Unsupported file format")

                if isinstance(data, pd.DataFrame):
                    if not data.select_dtypes(include=[np.number]).shape[1]:
                        raise ValueError("No numeric data found in the file")

                try:
                    self.matrix_data = data.astype(np.complex128)
                except ValueError as ve:
                    raise ValueError(f"Could not convert data to complex numbers: {ve}")

                self.rows = np.ones(self.matrix_data.shape[0], dtype=np.uint64)
                self.cols = np.ones(self.matrix_data.shape[1], dtype=np.uint64)

                self.matrix_display.setText(
                    f"Matrix Shape: {self.matrix_data.shape}\n\n{str(self.matrix_data)}"
                )

                self.status_label.setText(f"File imported successfully: {file_path}")
                self.statusBar.showMessage(
                    f"Matrix loaded: {self.matrix_data.shape[0]}x{self.matrix_data.shape[1]}"
                )

                self.calculate_button.setEnabled(True)

            except Exception as e:
                self.status_label.setText(f"Error importing file: {str(e)}")
                self.statusBar.showMessage("Error importing file")
                self.clear_data()
                self.calculate_button.setEnabled(False)

    def calculate(self):
        if self.matrix_data is None:
            self.status_label.setText("Please import a matrix first")
            return

        try:
            jax_matrix = jnp.array(self.matrix_data, dtype=jnp.complex128)
            jax_rows = jnp.array(self.rows, dtype=jnp.uint64)
            jax_cols = jnp.array(self.cols, dtype=jnp.uint64)

            permanent_value = perm(jax_matrix, jax_rows, jax_cols)

            gradient = jax.grad(perm, holomorphic=True)(jax_matrix, jax_rows, jax_cols)

            results_text = f"Permanent Value:\n{permanent_value}\n\n"
            results_text += f"Gradient Shape: {gradient.shape}\n\n"
            results_text += f"Gradient:\n{gradient}"

            self.results_display.setText(results_text)
            self.statusBar.showMessage("Calculation completed")

        except Exception as e:
            self.results_display.setText(f"Error calculating permanent: {str(e)}")
            self.statusBar.showMessage("Calculation error")

    def clear_data(self):
        self.matrix_data = None
        self.rows = None
        self.cols = None
        self.matrix_display.clear()
        self.results_display.clear()
        self.status_label.setText("Import a file to calculate the permanent")
        self.statusBar.showMessage("Ready")
        self.calculate_button.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = PermanentCalculatorUI()
    main_window.show()
    sys.exit(app.exec_())
