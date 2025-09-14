import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QMessageBox, QHBoxLayout
)
from ortools.sat.python import cp_model


# ================= BACKEND (your code reused) =================
def read_blocks(p):
    with open(p, 'r', encoding='utf-8') as f:
        next(f)
        blocks = []
        for r in csv.reader(f):
            if len(r) >= 2 and r[0].strip():
                inst_count = int(r[1].strip())
                if inst_count <= 0:
                    raise ValueError(f"Non-positive instance count for block {r[0]}")
                blocks.append((r[0].strip(), inst_count))
        if not blocks:
            raise ValueError("No valid blocks found")
        return blocks


def assign_blocks(blocks, num_licenses):
    model = cp_model.CpModel()
    blocks = sorted(blocks, key=lambda x: x[1], reverse=True)
    block_vars = {}
    for b, _ in blocks:
        block_vars[b] = [model.NewBoolVar(f'{b}_L{i+1}') for i in range(num_licenses)]
        model.Add(sum(block_vars[b]) == 1)
    for idx, (b, _) in enumerate(blocks[:num_licenses]):
        model.Add(block_vars[b][idx] == 1)
    totals = []
    for i in range(num_licenses):
        total = sum(block_vars[b][i] * c for b, c in blocks)
        totals.append(total)
    makespan = model.NewIntVar(0, sum(c for _, c in blocks), 'makespan')
    model.AddMaxEquality(makespan, totals)
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = []
        for b, _ in blocks:
            for i in range(num_licenses):
                if solver.Value(block_vars[b][i]):
                    schedule.append({'block': b, 'license': f'L{i+1}'})
                    break
        return schedule, solver.Value(makespan)
    return None, None


def write_schedule(schedule, p, num_licenses):
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for i in range(num_licenses):
            lic = [t for t in schedule if t['license'] == f'L{i+1}']
            w.writerow([f'License{i+1}'])
            w.writerow(['S.NO', 'Block'])
            for j, t in enumerate(lic, 1):
                w.writerow([j, t['block']])
            w.writerow([])


# ================= FRONTEND (PyQt5) =================
class SchedulerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Block Scheduler")
        self.resize(600, 400)

        self.blocks = []
        self.schedule = []
        self.num_licenses = 2

        layout = QVBoxLayout()

        # File selection
        self.file_label = QLabel("Input File: Not Selected")
        self.file_btn = QPushButton("Load CSV")
        self.file_btn.clicked.connect(self.load_file)

        # Licenses input
        hlayout = QHBoxLayout()
        self.license_label = QLabel("Number of Licenses:")
        self.license_input = QLineEdit("2")
        hlayout.addWidget(self.license_label)
        hlayout.addWidget(self.license_input)

        # Run button
        self.run_btn = QPushButton("Run Scheduler")
        self.run_btn.clicked.connect(self.run_scheduler)

        # Table for results
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Block", "License"])

        # Export button
        self.export_btn = QPushButton("Export Schedule to CSV")
        self.export_btn.clicked.connect(self.export_schedule)

        layout.addWidget(self.file_label)
        layout.addWidget(self.file_btn)
        layout.addLayout(hlayout)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.table)
        layout.addWidget(self.export_btn)

        self.setLayout(layout)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.blocks = read_blocks(path)
                self.file_label.setText(f"Input File: {path}")
                QMessageBox.information(self, "Success", f"Loaded {len(self.blocks)} blocks")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def run_scheduler(self):
        if not self.blocks:
            QMessageBox.warning(self, "Error", "Please load a valid CSV first.")
            return
        try:
            self.num_licenses = int(self.license_input.text())
            if self.num_licenses < 1:
                raise ValueError("Licenses must be positive")
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid number of licenses")
            return

        schedule, makespan = assign_blocks(self.blocks, self.num_licenses)
        if not schedule:
            QMessageBox.critical(self, "Error", "No feasible assignment found")
            return
        self.schedule = schedule
        self.table.setRowCount(len(schedule))
        for i, s in enumerate(schedule):
            self.table.setItem(i, 0, QTableWidgetItem(s['block']))
            self.table.setItem(i, 1, QTableWidgetItem(s['license']))
        QMessageBox.information(self, "Done", f"Schedule created. Makespan: {makespan}")

    def export_schedule(self):
        if not self.schedule:
            QMessageBox.warning(self, "Error", "No schedule to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Schedule", "schedule.csv", "CSV Files (*.csv)")
        if path:
            try:
                write_schedule(self.schedule, path, self.num_licenses)
                QMessageBox.information(self, "Success", f"Schedule saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SchedulerApp()
    win.show()
    sys.exit(app.exec_())
