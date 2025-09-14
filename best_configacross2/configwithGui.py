import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QFileDialog, QLabel, QTableWidget,
    QTableWidgetItem, QMessageBox
)

# ----------------- Metric Preferences -----------------
metric_pref = {
    'Total Power': 'high',
    'Overflow(H%)': 'low',
    'Overflow(V%)': 'low',
    'Max Hotspot': 'low',
    'Total Hotspot': 'low',
    'WNS': 'low',
    'TNS': 'low',
    'Unplaced Cells': 'low',
    'DRC Violations': 'low',
    'Density': 'low'
}

# ----------------- Evaluation Logic -----------------
def evaluate_best_config(df):
    # Normalize column names
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Identify config columns
    config_cols = [col for col in df.columns if col.lower().startswith('config')]
    if not config_cols:
        return None, "No configuration columns found."

    # Clean numeric values
    for col in config_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(':', '').str.replace(',', ''),
            errors='coerce'
        )
    df.dropna(inplace=True)

    # Comparison logic
    best_config = config_cols[0]
    for col in config_cols[1:]:
        better = 0
        for metric, direction in metric_pref.items():
            if metric in df['Parameter'].values:
                val_best = df.loc[df['Parameter'] == metric, best_config].values[0]
                val_curr = df.loc[df['Parameter'] == metric, col].values[0]
                if direction == 'high' and val_curr > val_best:
                    better += 1
                elif direction == 'low' and val_curr < val_best:
                    better += 1
        if better > len(metric_pref) // 2:
            best_config = col

    return best_config, None


# ----------------- PyQt Application -----------------
class BlockModelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLOCK_MODEL Config Comparator")
        self.setGeometry(200, 200, 800, 600)

        # Widgets
        self.label = QLabel("Upload a CSV file to begin.", self)
        self.label.setStyleSheet("font-size: 16px;")

        self.upload_btn = QPushButton("Upload CSV", self)
        self.upload_btn.clicked.connect(self.load_csv)

        self.result_label = QLabel("", self)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")

        self.table = QTableWidget(self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.result_label)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)", options=options
        )
        if file_name:
            try:
                df = pd.read_csv(file_name)

                # Evaluate best config
                best_config, error = evaluate_best_config(df)
                if error:
                    QMessageBox.critical(self, "Error", error)
                    return

                self.result_label.setText(f"✅ Best configuration: {best_config}")

                # Show table in UI
                self.populate_table(df)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read file: {e}")

    def populate_table(self, df):
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))


# ----------------- Main -----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlockModelApp()
    window.show()
    sys.exit(app.exec_())
