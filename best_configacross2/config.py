import pandas as pd

csv_file = "  "  #input csv 
def main(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = [col.strip().capitalize() for col in df.columns]

    config_cols = [col for col in df.columns if col.lower().startswith('config')]
    if not config_cols:
        print("No configuration columns found.")
        return

    for col in config_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(':','').str.replace(',',''), errors='coerce')

    df.dropna(inplace=True)

    if len(config_cols) == 1:
        print(f"Best configuration: {config_cols[0]}")
        return

    metric_pref = {
        'Total Power': 'high',
        'Unplaced Cells': 'low',
        'Overflow(H%)': 'low',
        'Overflow(V%)': 'low',
        'Max Hotspot': 'low',
        'Total Hotspot': 'low',
        'Violations': 'low'
    }

    best_config = config_cols[0]
    for col in config_cols[1:]:
        better = 0
        for metric, direction in metric_pref.items():
            if metric in df['Parameter'].values:
                val_best = df.loc[df['Parameter']==metric, best_config].values[0]
                val_curr = df.loc[df['Parameter']==metric, col].values[0]
                if direction == 'high' and val_curr > val_best:
                    better += 1
                elif direction == 'low' and val_curr < val_best:
                    better += 1
        if better > len(metric_pref)//2:
            best_config = col

    print(f"Best configuration: {best_config}")

if __name__ == "__main__":
    main(csv_file)