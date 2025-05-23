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
    QLineEdit,
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
from sooki import perm


class PermanentCalculatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Permanent Calculator")
        self.setGeometry(100, 100, 800, 600)
        self.matrix_data = None
        self.rows = None
        self.cols = None
        self.gradient_data = None

        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)

        import_action = file_menu.addAction("Import File")
        import_action.triggered.connect(self.import_file)

        self.save_gradient_action = file_menu.addAction("Save Gradient (.npy)")
        self.save_gradient_action.triggered.connect(self.save_gradient)
        self.save_gradient_action.setEnabled(False)

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

        self.rows_edit = QLineEdit()
        self.rows_edit.setPlaceholderText("Edit rows (comma-separated)")
        self.rows_edit.textChanged.connect(self.update_rows_from_edit)
        matrix_layout.addWidget(QLabel("Row multiplicities:"))
        matrix_layout.addWidget(self.rows_edit)

        self.cols_edit = QLineEdit()
        self.cols_edit.setPlaceholderText("Edit cols (comma-separated)")
        self.cols_edit.textChanged.connect(self.update_cols_from_edit)
        matrix_layout.addWidget(QLabel("Column multiplicities:"))
        matrix_layout.addWidget(self.cols_edit)

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
            "NumPy Files (*.npy);;CSV Files (*.csv);;All Files (*)",
            options=options,
        )

        if file_path:
            try:
                ext = file_path.lower()
                if ext.endswith(".npy"):
                    data = np.load(file_path, allow_pickle=True)
                    if isinstance(data, np.lib.npyio.NpzFile):
                        self.statusBar.showMessage("NPZ files are not supported here.")
                        self.status_label.setText("NPZ files are not supported here.")
                        self.clear_data()
                        self.calculate_button.setEnabled(False)
                        self.save_gradient_action.setEnabled(False)
                        return
                    if hasattr(data, "item") and isinstance(data.item(), dict):
                        data_dict = data.item()
                        matrix = data_dict.get("matrix")
                        rows = data_dict.get("rows")
                        cols = data_dict.get("cols")
                        if matrix is None or rows is None or cols is None:
                            self.statusBar.showMessage(
                                "The .npy file must contain 'matrix', 'rows', and 'cols' keys."
                            )
                            self.status_label.setText("The .npy file must contain 'matrix', 'rows', and 'cols' keys.")
                            self.clear_data()
                            self.calculate_button.setEnabled(False)
                            self.save_gradient_action.setEnabled(False)
                            return
                        self.matrix_data = np.array(matrix, dtype=np.complex128)
                        self.rows = np.array(rows, dtype=np.uint64)
                        self.cols = np.array(cols, dtype=np.uint64)
                        matrix_str = f"Matrix Shape: {self.matrix_data.shape}\n\n{str(self.matrix_data)}"
                        rows_str = f"Rows: {self.rows}"
                        cols_str = f"Cols: {self.cols}"
                        self.matrix_display.setText(
                            f"{matrix_str}\n\n{rows_str}\n{cols_str}"
                        )
                        self.rows_edit.setText(",".join(str(x) for x in self.rows))
                        self.cols_edit.setText(",".join(str(x) for x in self.cols))
                    else:
                        data = np.array(data)
                        if not isinstance(data, np.ndarray):
                            self.statusBar.showMessage(
                                "Loaded .npy file does not contain a NumPy array."
                            )
                            self.status_label.setText("Loaded .npy file does not contain a NumPy array.")
                            self.clear_data()
                            self.calculate_button.setEnabled(False)
                            self.save_gradient_action.setEnabled(False)
                            return
                        if data.ndim != 2:
                            self.statusBar.showMessage(
                                f"Expected a 2D array from .npy file, but got shape {data.shape}"
                            )
                            self.status_label.setText(f"Expected a 2D array from .npy file, but got shape {data.shape}")
                            self.clear_data()
                            self.calculate_button.setEnabled(False)
                            self.save_gradient_action.setEnabled(False)
                            return
                        self.matrix_data = data.astype(np.complex128)
                        self.rows = np.ones(self.matrix_data.shape[0], dtype=np.uint64)
                        self.cols = np.ones(self.matrix_data.shape[1], dtype=np.uint64)
                        self.matrix_display.setText(
                            f"Matrix Shape: {self.matrix_data.shape}\n\n{str(self.matrix_data)}"
                        )
                        self.rows_edit.setText(",".join(str(x) for x in self.rows))
                        self.cols_edit.setText(",".join(str(x) for x in self.cols))
                elif ext.endswith(".csv"):
                    data = pd.read_csv(file_path, header=None).to_numpy()
                    self.matrix_data = data.astype(np.complex128)
                    self.rows = np.ones(self.matrix_data.shape[0], dtype=np.uint64)
                    self.cols = np.ones(self.matrix_data.shape[1], dtype=np.uint64)
                    self.matrix_display.setText(
                        f"Matrix Shape: {self.matrix_data.shape}\n\n{str(self.matrix_data)}"
                    )
                    self.rows_edit.setText(",".join(str(x) for x in self.rows))
                    self.cols_edit.setText(",".join(str(x) for x in self.cols))
                elif ext.endswith(".xlsx"):
                    data = pd.read_excel(file_path, header=None).to_numpy()
                    self.matrix_data = data.astype(np.complex128)
                    self.rows = np.ones(self.matrix_data.shape[0], dtype=np.uint64)
                    self.cols = np.ones(self.matrix_data.shape[1], dtype=np.uint64)
                    self.matrix_display.setText(
                        f"Matrix Shape: {self.matrix_data.shape}\n\n{str(self.matrix_data)}"
                    )
                    self.rows_edit.setText(",".join(str(x) for x in self.rows))
                    self.cols_edit.setText(",".join(str(x) for x in self.cols))
                else:
                    self.statusBar.showMessage("Unsupported file format")
                    self.status_label.setText("Unsupported file format")
                    self.clear_data()
                    self.calculate_button.setEnabled(False)
                    self.save_gradient_action.setEnabled(False)
                    return
                self.status_label.setText(f"File imported successfully: {file_path}")
                self.statusBar.showMessage(
                    f"Matrix loaded: {self.matrix_data.shape[0]}x{self.matrix_data.shape[1]}"
                )

                self.calculate_button.setEnabled(True)
                self.gradient_data = None
                self.validate_multiplicities()

            except Exception as e:
                if self.statusBar.currentMessage() == "Ready":
                    self.statusBar.showMessage("Error importing file")
                self.status_label.setText(f"Error importing file: {str(e)}")
                self.clear_data()
                self.calculate_button.setEnabled(False)
                self.save_gradient_action.setEnabled(False)

    def save_gradient(self):
        if self.gradient_data is None:
            self.statusBar.showMessage("No gradient data to save.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Gradient",
            "",
            "NumPy Files (*.npy);;All Files (*)",
            options=options,
        )

        if file_path:
            if not file_path.endswith(".npy"):
                file_path += ".npy"
            try:
                np.save(file_path, self.gradient_data)
                self.statusBar.showMessage(
                    f"Gradient saved successfully to {file_path}"
                )
            except Exception as e:
                self.statusBar.showMessage(f"Error saving gradient: {str(e)}")

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
            self.gradient_data = np.array(gradient)

            results_text = f"Permanent Value:\n{permanent_value}\n\n"
            results_text += f"Gradient Shape: {gradient.shape}\n\n"
            results_text += f"Gradient:\n{gradient}"

            self.results_display.setText(results_text)
            self.statusBar.showMessage("Calculation completed")
            self.save_gradient_action.setEnabled(True)

        except Exception as e:
            self.results_display.setText(f"Error calculating permanent: {str(e)}")
            self.statusBar.showMessage("Calculation error")

    def clear_data(self):
        self.matrix_data = None
        self.rows = None
        self.cols = None
        self.matrix_display.clear()
        self.results_display.clear()
        self.save_gradient_action.setEnabled(False)
        # self.status_label.setText("Import a file to calculate the permanent")
        # self.statusBar.showMessage("Ready")
        self.calculate_button.setEnabled(False)
        self.rows_edit.clear()
        self.cols_edit.clear()

    def update_rows_from_edit(self):
        text = self.rows_edit.text()
        try:
            values = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
            if (
                self.matrix_data is not None
                and len(values) != self.matrix_data.shape[0]
            ):
                raise ValueError("Number of rows values does not match matrix size.")
            self.rows = np.array(values, dtype=np.uint64)
            matrix_str = (
                f"Matrix Shape: {self.matrix_data.shape}\n\n{str(self.matrix_data)}"
            )
            rows_str = f"Rows: {self.rows}"
            cols_str = f"Cols: {self.cols}"
            self.matrix_display.setText(f"{matrix_str}\n\n{rows_str}\n{cols_str}")
            self.validate_multiplicities()
        except Exception as e:
            self.statusBar.showMessage(f"Invalid rows input: {e}")
            self.calculate_button.setEnabled(False)

    def update_cols_from_edit(self):
        text = self.cols_edit.text()
        try:
            values = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
            if (
                self.matrix_data is not None
                and len(values) != self.matrix_data.shape[1]
            ):
                raise ValueError("Number of cols values does not match matrix size.")
            self.cols = np.array(values, dtype=np.uint64)
            matrix_str = (
                f"Matrix Shape: {self.matrix_data.shape}\n\n{str(self.matrix_data)}"
            )
            rows_str = f"Rows: {self.rows}"
            cols_str = f"Cols: {self.cols}"
            self.matrix_display.setText(f"{matrix_str}\n\n{rows_str}\n{cols_str}")
            self.validate_multiplicities()
        except Exception as e:
            self.statusBar.showMessage(f"Invalid cols input: {e}")
            self.calculate_button.setEnabled(False)

    def validate_multiplicities(self):
        if self.matrix_data is None or self.rows is None or self.cols is None:
            self.calculate_button.setEnabled(False)
            return
        if (
            len(self.rows) != self.matrix_data.shape[0]
            or len(self.cols) != self.matrix_data.shape[1]
        ):
            self.statusBar.showMessage(
                "Row/col vector length does not match matrix shape."
            )
            self.calculate_button.setEnabled(False)
            return
        if self.rows.sum() != self.cols.sum():
            self.statusBar.showMessage(
                f"Sum of rows ({self.rows.sum()}) does not match sum of cols ({self.cols.sum()})"
            )
            self.calculate_button.setEnabled(False)
            return
        self.calculate_button.setEnabled(True)
        self.statusBar.showMessage("Ready")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = PermanentCalculatorUI()
    main_window.show()
    sys.exit(app.exec_())
