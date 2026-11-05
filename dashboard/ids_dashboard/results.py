"""Measured experimental results copied from the saved outputs of ``main-h-mooa.ipynb``.

Every table records the notebook step it was taken from. Nothing here is
estimated or simulated: if a value was not printed by the notebook, it is not
included.
"""

import pandas as pd

PROJECT_TITLE = (
    "Algorithmic Optimization for Intrusion Detection: Multi-Objective Feature "
    "Selection Using Hybrid Global-Local Optimization"
)
DASHBOARD_TITLE = "Hybrid Optimization-Based Intrusion Detection System"
DATASET_NAME = "CIC DataSenseIoT 2025 (CIC-IIoT-2025)"
ORIGINAL_FEATURES = 71
N_CLASSES = 8
CLASS_NAMES = ["benign", "bruteforce", "ddos", "dos", "malware", "mitm", "recon", "web"]

TOTAL_RECORDS = 685_671
TEST_SAMPLES = 109_771

SOURCE_FINAL = "Notebook Step 6.55 (final held-out test consolidation)"
SOURCE_TIMING = "Notebook Step 6.56 (computational time integration)"
SOURCE_EDA = "Notebook Phase 4, Steps 4.4-4.5 (class distribution)"
SOURCE_SEEDS = "Notebook Steps 6.49-6.51 (five-seed robustness)"
SOURCE_SMOKE = "Notebook Step 6.64 (H-MOOA candidate evaluation smoke test)"
SOURCE_FPSO = "Notebook Step 6.65 (Binary Firefly-PSO hybrid)"
SOURCE_CUCKOO = "Notebook Step 6.66 (Cuckoo Search for RF hyperparameters)"
SOURCE_THRESH = "Notebook Step 6.67 (adaptive decision thresholds)"
SOURCE_IMPORTANCE = "Notebook Step 6.20 (six-feature RF importance)"

CLASS_PERCENTAGES = pd.DataFrame(
    {
        "Class": ["benign", "recon", "dos", "ddos", "mitm", "malware", "web", "bruteforce"],
        "Percentage (%)": [58.44, 15.44, 8.42, 8.27, 3.72, 3.53, 1.32, 0.88],
    }
)
LARGEST_CLASS = ("benign", 400_672)
SMALLEST_CLASS = ("bruteforce", 6_016)
IMBALANCE_RATIO = 66.60

RF_71_FEATURES = [
    "log_data-ranges_avg", "log_data-ranges_max", "log_data-ranges_min",
    "log_data-ranges_std_deviation", "log_data-types_count", "log_interval-messages",
    "log_messages_count", "network_fragmentation-score", "network_fragmented-packets",
    "network_header-length_avg", "network_header-length_max", "network_header-length_min",
    "network_header-length_std_deviation", "network_interval-packets", "network_ip-flags_avg",
    "network_ip-flags_max", "network_ip-flags_min", "network_ip-flags_std_deviation",
    "network_ip-length_avg", "network_ip-length_max", "network_ip-length_min",
    "network_ip-length_std_deviation", "network_ips_all_count", "network_ips_dst_count",
    "network_ips_src_count", "network_macs_all_count", "network_macs_dst_count",
    "network_macs_src_count", "network_mss_avg", "network_mss_max", "network_mss_min",
    "network_mss_std_deviation", "network_packet-size_avg", "network_packet-size_max",
    "network_packet-size_min", "network_packet-size_std_deviation", "network_packets_all_count",
    "network_packets_dst_count", "network_packets_src_count", "network_payload-length_avg",
    "network_payload-length_max", "network_payload-length_min",
    "network_payload-length_std_deviation", "network_ports_all_count", "network_ports_dst_count",
    "network_ports_src_count", "network_protocols_all_count", "network_protocols_dst_count",
    "network_protocols_src_count", "network_tcp-flags-ack_count", "network_tcp-flags-fin_count",
    "network_tcp-flags-psh_count", "network_tcp-flags-rst_count", "network_tcp-flags-syn_count",
    "network_tcp-flags-urg_count", "network_tcp-flags_avg", "network_tcp-flags_max",
    "network_tcp-flags_min", "network_tcp-flags_std_deviation", "network_time-delta_avg",
    "network_time-delta_max", "network_time-delta_min", "network_time-delta_std_deviation",
    "network_ttl_avg", "network_ttl_max", "network_ttl_min", "network_ttl_std_deviation",
    "network_window-size_avg", "network_window-size_max", "network_window-size_min",
    "network_window-size_std_deviation",
]

RF_9_FEATURES = [
    "network_interval-packets", "network_ip-flags_avg", "network_ips_all_count",
    "network_packets_all_count", "network_packets_dst_count", "network_time-delta_avg",
    "network_time-delta_min", "network_ttl_min", "network_window-size_std_deviation",
]

RF_6_FEATURES = [
    "network_ip-flags_avg", "network_ips_all_count", "network_time-delta_avg",
    "network_time-delta_min", "network_ttl_min", "network_window-size_std_deviation",
]

FINAL_TEST_METRICS = pd.DataFrame(
    {
        "Model": ["RF_71", "RF_9", "RF_6"],
        "Feature Count": [71, 9, 6],
        "Feature Reduction (%)": [0.0, 87.323944, 91.549296],
        "Accuracy": [0.970985, 0.963196, 0.958259],
        "Macro Precision": [0.977050, 0.965122, 0.956392],
        "Macro Recall": [0.964755, 0.951020, 0.945304],
        "Macro F1": [0.970803, 0.957948, 0.950784],
        "Weighted F1": [0.970935, 0.963110, 0.958172],
        "Macro FPR": [0.005999, 0.007284, 0.008072],
        "Training Time (s)": [54.550686, 39.713656, 27.967815],
        "Batch Prediction Time (s)": [0.625167, 0.490895, 0.514382],
    }
)
FINAL_TEST_METRICS["Latency (ms / 1,000 samples)"] = (
    FINAL_TEST_METRICS["Batch Prediction Time (s)"] / TEST_SAMPLES * 1_000_000
)

FINAL_RF_CONFIG = {
    "n_estimators": 100,
    "class_weight": "balanced",
    "random_state": 42,
}

CLASSWISE_F1 = pd.DataFrame(
    {
        "Class": CLASS_NAMES,
        "RF_71": [0.975604, 0.957232, 0.975751, 0.983700, 0.975444, 0.981366, 0.943901, 0.973424],
        "RF_9": [0.972973, 0.939189, 0.962908, 0.973964, 0.959329, 0.952848, 0.932375, 0.969997],
        "RF_6": [0.970155, 0.915825, 0.958333, 0.966073, 0.952543, 0.949069, 0.923444, 0.970832],
    }
)

ATTACK_TO_BENIGN = pd.DataFrame(
    {
        "Model": ["RF_71", "RF_9", "RF_6"],
        "Total Attack Samples": [53680, 53680, 53680],
        "Attacks Classified as Benign": [1772, 1955, 2036],
        "Attack-to-Benign Rate (%)": [3.3010, 3.6420, 3.7928],
    }
)

BOOTSTRAP = pd.DataFrame(
    {
        "Comparison": ["RF_71 - RF_9", "RF_71 - RF_9", "RF_71 - RF_6", "RF_71 - RF_6", "RF_9 - RF_6", "RF_9 - RF_6"],
        "Metric": ["Macro F1 Difference", "Macro FPR Difference"] * 3,
        "Mean Difference": [0.012871, -0.001289, 0.020085, -0.002073, 0.007174, -0.000789],
        "95% CI Lower": [0.011275, -0.001431, 0.018265, -0.002222, 0.005576, -0.000897],
        "95% CI Upper": [0.014396, -0.001151, 0.022073, -0.001913, 0.008805, -0.000669],
        "CI Excludes Zero": [True] * 6,
    }
)

MCNEMAR = pd.DataFrame(
    {
        "Comparison": ["RF_71 vs RF_9", "RF_71 vs RF_6", "RF_9 vs RF_6"],
        "A Correct, B Wrong": [1401, 2064, 1006],
        "A Wrong, B Correct": [546, 667, 464],
        "Log10 Holm-adjusted p-value": [-85.257312, -163.574129, -45.517327],
        "Significant at 0.05": [True, True, True],
    }
)

RF6_IMPORTANCE = pd.DataFrame(
    {
        "Feature": [
            "network_time-delta_avg", "network_time-delta_min", "network_window-size_std_deviation",
            "network_ip-flags_avg", "network_ttl_min", "network_ips_all_count",
        ],
        "Importance": [0.243295, 0.183980, 0.162233, 0.149691, 0.130565, 0.130236],
    }
)

FIVE_SEED_BEST = pd.DataFrame(
    {
        "Seed": [101, 202, 303, 404, 505],
        "Generation": [2, 1, 1, 1, 1],
        "Feature Count": [10, 9, 9, 9, 9],
        "Feature Reduction (%)": [85.915493, 87.323944, 87.323944, 87.323944, 87.323944],
        "Macro F1": [0.933268, 0.924307, 0.925396, 0.927103, 0.924766],
        "Macro FPR": [0.013277, 0.013732, 0.013728, 0.013603, 0.013692],
        "Fitness": [0.070864, 0.073514, 0.072860, 0.071811, 0.073231],
        "Jaccard vs RF_9": [0.727273, 1.0, 1.0, 1.0, 1.0],
    }
)

FEATURE_FREQUENCY = pd.DataFrame(
    {
        "Feature": [
            "network_interval-packets", "network_ip-flags_avg", "network_ips_all_count",
            "network_packets_all_count", "network_packets_dst_count", "network_time-delta_avg",
            "network_time-delta_min", "network_window-size_std_deviation", "network_ttl_min",
            "network_tcp-flags_avg", "network_window-size_min",
        ],
        "Selected in (of 5 seeds)": [5, 5, 5, 5, 5, 5, 5, 5, 4, 1, 1],
    }
)
FEATURE_FREQUENCY["Selection Frequency (%)"] = FEATURE_FREQUENCY["Selected in (of 5 seeds)"] * 20.0

SMOKE_TEST = pd.DataFrame(
    {
        "Candidate": ["RF_71", "RF_9"],
        "Feature Count": [71, 9],
        "Macro F1": [0.920537, 0.898904],
        "Macro FPR": [0.014987, 0.016685],
        "Training Time (s)": [0.887390, 0.677954],
        "Latency (ms / 1,000 samples)": [6.524257, 6.065901],
    }
)

FPSO_CONFIG = {
    "Population size": 8,
    "Search iterations": 2,
    "Candidate features": 71,
    "Minimum selected features": 5,
    "Random Forest trees": 50,
    "PSO inertia (w)": 0.65,
    "Cognitive coefficient (c1)": 1.5,
    "Social coefficient (c2)": 1.5,
    "Firefly beta0": 1.0,
    "Firefly gamma": 2.0,
    "Training samples": 20_000,
    "Validation samples": 5_000,
    "Unique model evaluations": 24,
    "Final Pareto archive size": 11,
}
FPSO_INITIAL_COUNTS = [71, 9, 17, 26, 9, 24, 15, 16]
FPSO_ITERATIONS = pd.DataFrame(
    {
        "Iteration": [1, 2],
        "Unique evaluations": [16, 24],
        "Pareto archive size": [7, 11],
        "Population feature counts": [
            "35, 45, 47, 43, 45, 37, 39, 32",
            "33, 29, 45, 42, 40, 33, 37, 32",
        ],
    }
)
FPSO_ARCHIVE = pd.DataFrame(
    {
        "Candidate": list(range(11)),
        "Feature Count": [71, 9, 24, 45, 32, 33, 29, 45, 42, 40, 33],
        "Macro F1": [0.920537, 0.898904, 0.897796, 0.908814, 0.912726, 0.916322, 0.896801, 0.917179, 0.911159, 0.919910, 0.916087],
        "Macro FPR": [0.014987, 0.016685, 0.016073, 0.015698, 0.015917, 0.016235, 0.015925, 0.014951, 0.015064, 0.015276, 0.016170],
        "Feature Reduction (%)": [0.0, 87.323944, 66.197183, 36.619718, 54.929577, 53.521127, 59.154930, 36.619718, 40.845070, 43.661972, 53.521127],
        "Latency (ms / 1,000 samples)": [6.127698, 5.890274, 5.979144, 5.944598, 5.875980, 5.937793, 6.009680, 6.097054, 5.961170, 5.962107, 6.091636],
    }
)

CUCKOO_CONFIG = {
    "Number of nests": 6,
    "Iterations": 2,
    "Abandonment rate (pa)": 0.25,
    "Levy flight beta": 1.5,
    "Feature subset": "RF_9 (9 features)",
    "Training samples": 20_000,
    "Validation samples": 5_000,
    "Unique hyperparameter evaluations": 22,
    "Final Pareto archive size": 4,
}
CUCKOO_EVALUATIONS = pd.DataFrame(
    {
        "n_estimators": [75, 79, 84, 41, 65, 88, 76, 35, 69, 70, 54, 44, 84, 98, 93, 87, 47, 75, 50, 84, 74, 44],
        "max_depth": [9, 8, 30, 9, 8, 22, 30, 5, 5, 10, 13, 5, 25, 27, 25, 22, 16, 25, 15, 21, 11, 30],
        "min_samples_leaf": [7, 3, 6, 8, 6, 4, 6, 8, 6, 6, 6, 8, 5, 3, 6, 5, 6, 1, 6, 8, 6, 5],
        "max_features": ["log2", "sqrt", "sqrt", "0.5", "0.5", "0.5", "sqrt", "0.5", "0.5", "0.8", "0.5", "0.5",
                         "0.5", "log2", "0.5", "0.5", "0.5", "log2", "0.5", "0.5", "0.8", "0.8"],
        "Macro F1": [0.839683, 0.827724, 0.861344, 0.832244, 0.823669, 0.880519, 0.861074, 0.723443, 0.728891,
                     0.846048, 0.861336, 0.722772, 0.866020, 0.883156, 0.867283, 0.869309, 0.866339, 0.895622,
                     0.867515, 0.857725, 0.857121, 0.869482],
        "Macro FPR": [0.018763, 0.019370, 0.018045, 0.018797, 0.018875, 0.017105, 0.017963, 0.026226, 0.025590,
                      0.018284, 0.017789, 0.026321, 0.017834, 0.016956, 0.017619, 0.017578, 0.017634, 0.016880,
                      0.017527, 0.018123, 0.018016, 0.017299],
        "Training Time (s)": [0.781955, 0.755254, 0.947249, 0.524360, 0.771863, 1.271747, 0.921696, 0.351918,
                              0.643768, 1.400905, 0.737355, 0.429301, 1.162884, 1.136109, 1.266602, 1.196811,
                              0.676915, 0.934334, 0.722154, 1.142984, 1.443605, 0.987952],
        "Latency (ms / 1,000 samples)": [10.859989, 10.967702, 10.943443, 5.797397, 8.426799, 11.041856,
                                         8.503045, 5.889731, 8.572907, 8.420412, 8.424934, 5.877503, 10.939761,
                                         10.893827, 11.100521, 11.029648, 5.900811, 8.652372, 5.883445,
                                         10.996790, 8.382357, 5.887244],
    }
)
CUCKOO_EVALUATIONS.insert(0, "Evaluation", range(len(CUCKOO_EVALUATIONS)))
CUCKOO_EVALUATIONS["Stage"] = ["Initial nest"] * 6 + ["Search iteration"] * 16
CUCKOO_ARCHIVE_IDS = [3, 18, 17, 21]

THRESHOLDS = pd.DataFrame(
    {
        "Class": CLASS_NAMES,
        "Optimized Threshold": [0.50, 0.30, 0.55, 0.45, 0.50, 0.35, 0.15, 0.15],
    }
)
THRESHOLD_SPLIT = {
    "Training samples": 20_000,
    "Calibration samples": 2_500,
    "Assessment samples": 2_500,
    "Feature subset": "RF_9",
    "Calibration/assessment overlap": "None",
    "Objective": "(1 - Macro F1) + 2 x Macro FPR",
    "Threshold grid": "0.05 to 0.95, step 0.05",
}
THRESHOLD_ASSESSMENT = pd.DataFrame(
    {
        "Rule": ["Default argmax", "Adaptive thresholds"],
        "Accuracy": [0.9296, 0.9280],
        "Macro Precision": [0.927780, 0.927684],
        "Macro Recall": [0.868823, 0.866576],
        "Macro F1": [0.896299, 0.895185],
        "Macro FPR": [0.017573, 0.017565],
    }
)
THRESHOLD_CALIBRATION = pd.DataFrame(
    {
        "Rule": ["Default argmax", "Adaptive thresholds"],
        "Accuracy": [0.9324, 0.9368],
        "Macro Precision": [0.923319, 0.940891],
        "Macro Recall": [0.870683, 0.884792],
        "Macro F1": [0.895018, 0.911067],
        "Macro FPR": [0.016188, 0.015094],
    }
)
THRESHOLD_RECALL = pd.DataFrame(
    {
        "Class": CLASS_NAMES,
        "Support": [1496, 22, 199, 201, 85, 89, 376, 32],
        "Default Recall": [0.986631, 0.772727, 0.849246, 0.900498, 0.847059, 0.887640, 0.800532, 0.906250],
        "Adaptive Recall": [0.984626, 0.772727, 0.839196, 0.900498, 0.847059, 0.876404, 0.805851, 0.906250],
    }
)
THRESHOLD_RECALL["Recall Change"] = THRESHOLD_RECALL["Adaptive Recall"] - THRESHOLD_RECALL["Default Recall"]
