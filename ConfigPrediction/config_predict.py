
#predict the values using old data  

Input configs are 4 taken and train the model with their respective values and need to predict the data for the 5th config. 



## INPUTS:
# 1. CONFIG files (TCL files)
# 2. CSV files corresponding to each config (e.g., "final1.csv")  , i will attach csv format named "final1.csv"

##############################################################
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib  
tcl_files = ['config1.tcl', 'config2.tcl', 'config3.tcl', 'config4.tcl']
csv_files = ['final1.csv', 'final2.csv', 'final3.csv', 'final4.csv']
def parse_tcl_to_dict(tcl_file):
    data = {}
    with open(tcl_file, 'r') as f:
        for line in f:
            if line.strip().startswith("set "):
                parts = line.strip().split(maxsplit=2)
                if len(parts) == 3:
                    key = parts[1]
                    value = parts[2].strip('"')
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
        print(f"Error reading result file {csv_file}: {e}")
        return None

def build_dataset(tcl_files, csv_files):
    features = []
    targets = []

    for tcl_file, csv_file in zip(tcl_files, csv_files):
        try:
            feature_dict = parse_tcl_to_dict(tcl_file)
            if not feature_dict:
                print(f"Warning: No valid data in {tcl_file}")
                continue
            feature_df = pd.DataFrame([feature_dict])
            features.append(feature_df)
        except Exception as e:
            print(f"Error reading feature file {tcl_file}: {e}")

        target = parse_target_csv(csv_file)
        if target:
            targets.append(target)

    if not features or not targets:
        raise ValueError("No valid features or targets to build the dataset.")

    X = pd.concat(features, ignore_index=True)
    Y = pd.DataFrame(targets, columns=[
        "Total Power", "Overflow(%H)", "Overflow(%V)",
        "Max Hotspot", "Total Hotspot", "WNS", "TNS", "violations","density"
    ])
    return X, Y

if __name__ == "__main__":
    X, Y = build_dataset(tcl_files, csv_files)
    print("\nFeatures (X):")
    print(X)
    print("\nTargets (Y):")
    print(Y)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes

    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    
    joblib.dump(model, "rf_model.pkl")
    joblib.dump(list(X.columns), "model_features.pkl")         # Save feature names
    joblib.dump(list(Y.columns), "target_labels.pkl")          # Save target column names
    
    print("\nModel saved as 'rf_model.pkl'.")

    
    predictions = model.predict(X_test)
    print("\nPredictions:")
    print(predictions)

###########################################################
##after above code it generates a rf_model.pkl .  

 

 

##this code for the predict the values for the 5th config using 4 configs 

################################################
import pandas as pd
import joblib
import re

# Load model and feature list
model = joblib.load("rf_model.pkl")
print("Model loaded successfully.")

expected_features = joblib.load("model_features.pkl")
print("Feature list loaded successfully.")

config_files = ['config1.tcl', 'config2.tcl', 'config3.tcl', 'config4.tcl', 'config5.tcl']
config_names = [f'config{i+1}' for i in range(len(config_files))]

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

# Prepare features
all_features = []
for file in config_files:
    features = parse_tcl_to_dict(file)
    all_features.append(features)

df_features = pd.DataFrame(all_features)
df_features = pd.get_dummies(df_features)

# Align features with training
df_features = df_features.reindex(columns=expected_features, fill_value=0)

# Predict
predictions = model.predict(df_features)

# Format result
target_columns = [
    "Total Power", "Overflow(%H)", "Overflow(%V)",
    "Max Hotspot", "Total Hotspot", "WNS", "TNS", "violations", "density"
]

result_df = pd.DataFrame(predictions.T, index=target_columns, columns=config_names)
result_df.reset_index(inplace=True)
result_df.rename(columns={"index": "Parameter"}, inplace=True)
result_df.to_csv("predicted_outputs.csv", index=False)

print("Predictions saved to 'predicted_outputs.csv'.")

###########################################
