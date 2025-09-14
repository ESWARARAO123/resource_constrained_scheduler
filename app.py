import sys
import os
import joblib
import pandas as pd
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QTextEdit, QMessageBox, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import re


# ----------------------------
# BACKEND FUNCTIONS (Your original logic, unchanged)
# ----------------------------

def parse_tcl_to_dict(tcl_file):
    data = {}
    with open(tcl_file, 'r') as f:
        for line in f:
            match = re.match(r'^\s*set\s+(\S+)\s+"?([^"\n]+)"?', line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                data[key] = value
    return data

def parse_target_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        df.set_index("Parameter", inplace=True)
        result = df["run_1"].to_dict()
        return [
            float(result.get("Total Power", 0)),
            float(result.get("Overflow(%H)", 0)),
            float(result.get("Overflow(%V)", 0)),
            float(result.get("Max Hotspot", 0)),
            float(result.get("Total Hotspot", 0)),
            float(result.get("WNS", 0)),
            float(result.get("TNS", 0)),
            float(result.get("violations", 0)),
            float(result.get("density", 0))
        ]
    except Exception as e:
        raise Exception(f"Error reading {csv_file}: {e}")

def build_dataset_from_files(tcl_files, csv_files):
    features = []
    targets = []

    for tcl_file, csv_file in zip(tcl_files, csv_files):
        feature_dict = parse_tcl_to_dict(tcl_file)
        if feature_dict:
            features.append(feature_dict)
        target = parse_target_csv(csv_file)
        if target:
            targets.append(target)

    if not features or not targets:
        raise ValueError("No valid features or targets")

    X = pd.DataFrame(features)
    Y = pd.DataFrame(targets, columns=[
        "Total Power", "Overflow(%H)", "Overflow(%V)",
        "Max Hotspot", "Total Hotspot", "WNS", "TNS", "violations", "density"
    ])

    # Encode categorical features using cat.codes (same as training!)
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes

    return X, Y

def predict_config5(tcl_files):
    try:
        model = joblib.load("rf_model.pkl")
        expected_features = joblib.load("model_features.pkl")
        target_columns = joblib.load("target_labels.pkl")
    except FileNotFoundError:
        raise Exception("Model files not found. Train first!")

    all_features = []
    for tcl_file in tcl_files:
        features = parse_tcl_to_dict(tcl_file)
        all_features.append(features)

    df_features = pd.DataFrame(all_features)

    for col in df_features.columns:
        if df_features[col].dtype == "object":
            df_features[col] = df_features[col].astype("category").cat.codes

    df_features = df_features.reindex(columns=expected_features, fill_value=0)

    predictions = model.predict(df_features)

    result_df = pd.DataFrame(predictions.T, index=target_columns, columns=[f'config{i+1}' for i in range(len(tcl_files))])
    result_df.reset_index(inplace=True)
    result_df.rename(columns={"index": "Parameter"}, inplace=True)

    return result_df


# ----------------------------
# PYQT5 FRONTEND
# ----------------------------

class EDAPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EDA Config Predictor - Desktop App")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("font-family: Arial; font-size: 11pt;")

        # Store file paths
        self.tcl_files = [None] * 5
        self.csv_files = [None] * 4

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Title
        title = QLabel("🚀 EDA Configuration Predictor")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Instructions
        instr = QLabel(
            "Upload 5 TCL config files and 4 CSV result files to train the model and predict performance for config5."
        )
        instr.setWordWrap(True)
        instr.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(instr)

        # Separator
        main_layout.addSpacing(20)

        # --- UPLOAD SECTION ---
        upload_group = QGroupBox("📁 Upload Files")
        upload_layout = QVBoxLayout()
        upload_group.setLayout(upload_layout)

        # TCL Files
        tcl_label = QLabel("TCL Config Files (All 5)")
        tcl_label.setFont(QFont("Arial", 10, QFont.Bold))
        upload_layout.addWidget(tcl_label)

        self.tcl_buttons = []
        for i in range(1, 6):
            btn = QPushButton(f"Select config{i}.tcl")
            btn.clicked.connect(lambda _, idx=i-1: self.select_file(idx, "tcl"))
            upload_layout.addWidget(btn)
            self.tcl_buttons.append(btn)

        # CSV Files
        csv_label = QLabel("\nCSV Result Files (Configs 1–4 only)")
        csv_label.setFont(QFont("Arial", 10, QFont.Bold))
        upload_layout.addWidget(csv_label)

        self.csv_buttons = []
        for i in range(1, 5):
            btn = QPushButton(f"Select final{i}.csv")
            btn.clicked.connect(lambda _, idx=i-1: self.select_file(idx, "csv"))
            upload_layout.addWidget(btn)
            self.csv_buttons.append(btn)

        main_layout.addWidget(upload_group)

        # --- ACTION BUTTON ---
        self.predict_btn = QPushButton("⚡ Train Model & Predict config5")
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px; font-size: 14px;")
        self.predict_btn.clicked.connect(self.run_prediction)
        main_layout.addWidget(self.predict_btn)

        # --- RESULTS DISPLAY ---
        results_group = QGroupBox("📊 Predicted Results for config5")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)  # Parameter, config5
        self.result_table.setHorizontalHeaderLabels(["Parameter", "config5"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.result_table)

        self.download_btn = QPushButton("📥 Download Predictions as CSV")
        self.download_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.download_btn.clicked.connect(self.download_results)
        self.download_btn.setEnabled(False)
        results_layout.addWidget(self.download_btn)

        main_layout.addWidget(results_group)

        # --- STATUS LOG ---
        log_group = QGroupBox("📝 Status Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)

        main_layout.addWidget(log_group)

        # Footer
        footer = QLabel("Built with PyQt5 • All processing happens locally.")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray; font-size: 9pt;")
        main_layout.addWidget(footer)

    def select_file(self, idx, file_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {file_type.upper()} file",
            "",
            f"{file_type.upper()} Files (*.{file_type});;All Files (*)",
            options=options
        )

        if file_path:
            if file_type == "tcl":
                self.tcl_files[idx] = file_path
                self.tcl_buttons[idx].setText(f"✓ config{idx+1}.tcl")
            else:
                self.csv_files[idx] = file_path
                self.csv_buttons[idx].setText(f"✓ final{idx+1}.csv")
            self.log_text.append(f"✅ Selected: {os.path.basename(file_path)}")

    def run_prediction(self):
        # Validate uploads
        missing_tcl = [i+1 for i, f in enumerate(self.tcl_files) if f is None]
        missing_csv = [i+1 for i, f in enumerate(self.csv_files) if f is None]

        if missing_tcl:
            QMessageBox.critical(self, "Missing Files", f"Please select all TCL files: config{missing_tcl}.tcl")
            return
        if missing_csv:
            QMessageBox.critical(self, "Missing Files", f"Please select all CSV files: final{missing_csv}.csv")
            return

        self.log_text.append("\n🔄 Processing... This may take a moment...")
        self.predict_btn.setEnabled(False)
        self.download_btn.setEnabled(False)

        try:
            with TemporaryDirectory() as tmpdir:
                # Copy uploaded files to temp dir
                tcl_paths = []
                csv_paths = []

                for i, path in enumerate(self.tcl_files):
                    dest = os.path.join(tmpdir, f"config{i+1}.tcl")
                    with open(dest, 'wb') as f_out, open(path, 'rb') as f_in:
                        f_out.write(f_in.read())
                    tcl_paths.append(dest)

                for i, path in enumerate(self.csv_files):
                    dest = os.path.join(tmpdir, f"final{i+1}.csv")
                    with open(dest, 'wb') as f_out, open(path, 'rb') as f_in:
                        f_out.write(f_in.read())
                    csv_paths.append(dest)

                # Train model
                X, Y = build_dataset_from_files(tcl_paths, csv_paths)
                model = joblib.load("rf_model.pkl") if os.path.exists("rf_model.pkl") else None
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, Y)

                # Save model artifacts
                joblib.dump(model, "rf_model.pkl")
                joblib.dump(list(X.columns), "model_features.pkl")
                joblib.dump(list(Y.columns), "target_labels.pkl")

                self.log_text.append("✅ Model trained successfully!")

                # Predict config5
                predictions_df = predict_config5(tcl_paths)

                # Display in table
                self.result_table.setRowCount(len(predictions_df))
                for row in range(len(predictions_df)):
                    param = predictions_df.iloc[row]["Parameter"]
                    val = predictions_df.iloc[row]["config5"]
                    self.result_table.setItem(row, 0, QTableWidgetItem(str(param)))
                    self.result_table.setItem(row, 1, QTableWidgetItem(f"{val:.4f}"))

                self.log_text.append("🎯 Prediction complete!")
                self.download_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log_text.append(f"❌ Error: {str(e)}")

        finally:
            self.predict_btn.setEnabled(True)

    def download_results(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Predictions",
            "predicted_outputs.csv",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            if not file_path.endswith(".csv"):
                file_path += ".csv"

            # Get current table data
            headers = ["Parameter", "config5"]
            rows = []
            for row in range(self.result_table.rowCount()):
                param = self.result_table.item(row, 0).text()
                val = self.result_table.item(row, 1).text()
                rows.append([param, val])

            df = pd.DataFrame(rows, columns=headers)
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Saved", f"Results saved to:\n{file_path}")
            self.log_text.append(f"💾 Saved predictions to: {file_path}")


# ----------------------------
# MAIN ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EDAPredictorApp()
    window.show()
    sys.exit(app.exec_())