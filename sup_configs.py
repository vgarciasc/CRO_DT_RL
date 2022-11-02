import csv
import pdb

import numpy as np

config_BC = {
    "name": "Breast cancer",
    "filepath": "datasets/breast-cancer-wisconsin.data",
    "n_attributes": 9,
    "attributes": [
        "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
        "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
        "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"],
    "n_classes": 2,
    "classes": [(2, "Benign"), (4, "Malignant")]
}

config_CE = {
    "name": "Car evaluation",
    "filepath": "datasets/car.data",
    "n_attributes": 6,
    "attributes": ["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    "n_classes": 4,
    "classes": [(0, "Unacceptable"), (1, "Acceptable"), (2, "Good"), (3, "Very good")]
}

config_BN = {
    "name": "Banknote authentication",
    "filepath": "datasets/data_banknote_authentication.txt",
    "n_attributes": 4,
    "attributes": ["variance", "skewness", "curtosis", "entropy"],
    "n_classes": 4,
    "classes": [(0, "Authentic"), (1, "Forged")]
}

config_BS = {
    "name": "Balance scale",
    "filepath": "datasets/balance-scale.data",
    "n_attributes": 4,
    "attributes": ["left weight", "left distance", "right weight", "right distance"],
    "n_classes": 3,
    "classes": [(0, "Left"), (1, "Balanced"), (2, "Right")]
}

config_AI1 = {
    "name": "Acute inflammations 1",
    "filepath": "datasets/acute-inflammations-1.data",
    "n_attributes": 6,
    "attributes": ["temperature", "nausea", "lumbar pain", "urine pushing", "micturition", "burning urethra"],
    "n_classes": 2,
    "classes": [(0, "No inflammation"), (1, "Inflammation")]
}

config_AI2 = {
    "name": "Acute inflammations 2",
    "filepath": "datasets/acute-inflammations-2.data",
    "n_attributes": 6,
    "attributes": ["temperature", "nausea", "lumbar pain", "urine pushing", "micturition", "burning urethra"],
    "n_classes": 2,
    "classes": [(0, "No nephritis"), (1, "Nephritis")]
}

config_BT = {
    "name": "Blood transfusion",
    "filepath": "datasets/blood-transfusion.data",
    "n_attributes": 4,
    "attributes": ["recency", "frequency", "monetary", "time", "donated"],
    "n_classes": 2,
    "classes": [(0, "Not donor"), (1, "Donor")]
}

config_CC = {
    "name": "Climate model crashes",
    "filepath": "datasets/climate-crashes.data",
    "n_attributes": 18,
    "attributes": ["vconst_corr", "vconst_2", "vconst_3", "vconst_4", "vconst_5", "vconst_7", "ah_corr", "ah_bolus", "slm_corr", "efficiency_factor", "tidal_mix_max", "vertical_decay_scale", "convect_corr", "bckgrnd_vdc1", "bckgrnd_vdc_ban", "bckgrnd_vdc_eq", "bckgrnd_vdc_psim", "Prandtl"],
    "n_classes": 2,
    "classes": [(0, "Failure"), (1, "Success")]
}

config_CB = {
    "name": "Connectionist bench sonar",
    "filepath": "datasets/sonar.all-data",
    "n_attributes": 60,
    "attributes": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38", "x39", "x40", "x41", "x42", "x43", "x44", "x45", "x46", "x47", "x48", "x49", "x50", "x51", "x52", "x53", "x54", "x55", "x56", "x57", "x58", "x59", "x60"],
    "n_classes": 2,
    "classes": [(0, "Rock"), (1, "Mine")]
}

config_OC = {
    "name": "Optical recognition",
    "filepath": "datasets/optdigits.tra",
    "n_attributes": 64,
    "attributes": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38", "x39", "x40", "x41", "x42", "x43", "x44", "x45", "x46", "x47", "x48", "x49", "x50", "x51", "x52", "x53", "x54", "x55", "x56", "x57", "x58", "x59", "x60", "x61", "x62", "x63", "x64"],
    "n_classes": 10,
    "classes": [(0, "Number 0"), (1, "Number 1"), (2, "Number 2"), (3, "Number 3"), (4, "Number 4"), (5, "Number 5"), (6, "Number 6"), (7, "Number 7"), (8, "Number 8"), (9, "Number 9")]
}

def get_config(dataset_name):
    if dataset_name == "breast_cancer":
        return config_BC
    elif dataset_name == "car":
        return config_CE
    elif dataset_name == "banknote":
        return config_BN
    elif dataset_name == "balance":
        return config_BS
    elif dataset_name == "acute-1":
        return config_AI1
    elif dataset_name == "acute-2":
        return config_AI2
    elif dataset_name == "transfusion":
        return config_BT
    elif dataset_name == "climate":
        return config_CC
    elif dataset_name == "sonar":
        return config_CB
    elif dataset_name == "optical":
        return config_OC
        
    raise Exception(f"Invalid dataset_name {dataset_name}.")

def load_dataset(config):
    try:
        Xy = np.loadtxt(open(config["filepath"], "r"), delimiter=",")
    except:
        pdb.set_trace()
    X = Xy[:,:-1]
    y = Xy[:,-1]
    y = [[i for i, (c, _) in enumerate(config["classes"]) if c == y_i][0] for y_i in y]
    return np.array(X), np.array(y)

if __name__ == "__main__":
    pdb.set_trace()