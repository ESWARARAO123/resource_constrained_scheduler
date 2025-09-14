import csv, sys
from ortools.sat.python import cp_model

def read_blocks(p):
    try:
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
    except FileNotFoundError:
        print(f"Blocks file '{p}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{p}': {e}")
        sys.exit(1)

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
    print(f"Solver status: {solver.StatusName(status)}")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = []
        for b, _ in blocks:
            for i in range(num_licenses):
                if solver.Value(block_vars[b][i]):
                    schedule.append({'block': b, 'license': f'L{i+1}'})
                    break
        for i in range(num_licenses):
            total = sum(c for b, c in blocks if any(s['block'] == b and s['license'] == f'L{i+1}' for s in schedule))
            print(f"License L{i+1} total: {total}")
        print(f"Makespan: {solver.Value(makespan)}")
        return schedule
    print("No feasible assignment found")
    sys.exit(1)

def write_schedule(schedule, p, num_licenses):
    try:
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for i in range(num_licenses):
                lic = [t for t in schedule if t['license'] == f'L{i+1}']
                w.writerow([f'License{i+1}'])
                w.writerow(['S.NO', 'Block'])
                for j, t in enumerate(lic, 1):
                    w.writerow([j, t['block']])
                w.writerow([])
        print(f"Schedule written to '{p}'")
    except Exception as e:
        print(f"Error writing '{p}': {e}")
        sys.exit(1)

def main():
    num_licenses = 2
    if len(sys.argv) > 1:
        try:
            num_licenses = int(sys.argv[1])
            if num_licenses < 1:
                raise ValueError("Number of licenses must be positive")
        except ValueError as e:
            print(f"Invalid number of licenses: {e}")
            sys.exit(1)
    blocks = read_blocks("instance_count.csv")
    schedule = assign_blocks(blocks, num_licenses)
    write_schedule(schedule, "schedule.csv", num_licenses)

if __name__ == "__main__":
    main()

