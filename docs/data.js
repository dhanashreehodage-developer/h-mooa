// Measured results copied from saved outputs of main-h-mooa.ipynb (see step references).
const DATA = {
  classes: ["benign", "bruteforce", "ddos", "dos", "malware", "mitm", "recon", "web"],
  classDist: {
    labels: ["benign", "recon", "dos", "ddos", "mitm", "malware", "web", "bruteforce"],
    pct: [58.44, 15.44, 8.42, 8.27, 3.72, 3.53, 1.32, 0.88],
  },
  final: [
    { model: "RF_71", feats: 71, red: 0.0, acc: 0.970985, prec: 0.97705, rec: 0.964755, f1: 0.970803, wf1: 0.970935, fpr: 0.005999, train: 54.550686, pred: 0.625167 },
    { model: "RF_9", feats: 9, red: 87.323944, acc: 0.963196, prec: 0.965122, rec: 0.95102, f1: 0.957948, wf1: 0.96311, fpr: 0.007284, train: 39.713656, pred: 0.490895 },
    { model: "RF_6", feats: 6, red: 91.549296, acc: 0.958259, prec: 0.956392, rec: 0.945304, f1: 0.950784, wf1: 0.958172, fpr: 0.008072, train: 27.967815, pred: 0.514382 },
  ],
  classwise: {
    RF_71: [0.975604, 0.957232, 0.975751, 0.9837, 0.975444, 0.981366, 0.943901, 0.973424],
    RF_9: [0.972973, 0.939189, 0.962908, 0.973964, 0.959329, 0.952848, 0.932375, 0.969997],
    RF_6: [0.970155, 0.915825, 0.958333, 0.966073, 0.952543, 0.949069, 0.923444, 0.970832],
  },
  confusionRF9: [
    [54991, 22, 75, 54, 32, 30, 877, 10],
    [33, 973, 0, 4, 24, 8, 7, 0],
    [238, 6, 10293, 150, 15, 80, 57, 0],
    [130, 1, 115, 10942, 20, 13, 41, 2],
    [80, 6, 10, 13, 4517, 26, 74, 0],
    [119, 6, 14, 20, 31, 4668, 52, 0],
    [1287, 9, 33, 22, 46, 63, 17779, 0],
    [68, 0, 0, 0, 6, 0, 11, 1568],
  ],
  attackToBenign: [
    ["RF_71", 1772, 3.301],
    ["RF_9", 1955, 3.642],
    ["RF_6", 2036, 3.7928],
  ],
  rf9: ["network_interval-packets", "network_ip-flags_avg", "network_ips_all_count", "network_packets_all_count", "network_packets_dst_count", "network_time-delta_avg", "network_time-delta_min", "network_ttl_min", "network_window-size_std_deviation"],
  rf6: ["network_ip-flags_avg", "network_ips_all_count", "network_time-delta_avg", "network_time-delta_min", "network_ttl_min", "network_window-size_std_deviation"],
  importance: {
    labels: ["network_time-delta_avg", "network_time-delta_min", "network_window-size_std_deviation", "network_ip-flags_avg", "network_ttl_min", "network_ips_all_count"],
    values: [0.243295, 0.18398, 0.162233, 0.149691, 0.130565, 0.130236],
  },
  bootstrap: [
    ["RF_71 - RF_9", "Macro F1", 0.012871, 0.011275, 0.014396],
    ["RF_71 - RF_9", "Macro FPR", -0.001289, -0.001431, -0.001151],
    ["RF_71 - RF_6", "Macro F1", 0.020085, 0.018265, 0.022073],
    ["RF_71 - RF_6", "Macro FPR", -0.002073, -0.002222, -0.001913],
    ["RF_9 - RF_6", "Macro F1", 0.007174, 0.005576, 0.008805],
    ["RF_9 - RF_6", "Macro FPR", -0.000789, -0.000897, -0.000669],
  ],
  mcnemar: [
    ["RF_71 vs RF_9", 1401, 546, -85.257312],
    ["RF_71 vs RF_6", 2064, 667, -163.574129],
    ["RF_9 vs RF_6", 1006, 464, -45.517327],
  ],
  seeds: [
    [101, 10, 0.933268, 0.013277, 0.727273],
    [202, 9, 0.924307, 0.013732, 1.0],
    [303, 9, 0.925396, 0.013728, 1.0],
    [404, 9, 0.927103, 0.013603, 1.0],
    [505, 9, 0.924766, 0.013692, 1.0],
  ],
  freq: {
    labels: ["network_interval-packets", "network_ip-flags_avg", "network_ips_all_count", "network_packets_all_count", "network_packets_dst_count", "network_time-delta_avg", "network_time-delta_min", "network_window-size_std_deviation", "network_ttl_min", "network_tcp-flags_avg", "network_window-size_min"],
    values: [5, 5, 5, 5, 5, 5, 5, 5, 4, 1, 1],
  },
  fpsoArchive: {
    feats: [71, 9, 24, 45, 32, 33, 29, 45, 42, 40, 33],
    f1: [0.920537, 0.898904, 0.897796, 0.908814, 0.912726, 0.916322, 0.896801, 0.917179, 0.911159, 0.91991, 0.916087],
    fpr: [0.014987, 0.016685, 0.016073, 0.015698, 0.015917, 0.016235, 0.015925, 0.014951, 0.015064, 0.015276, 0.01617],
  },
  cuckoo: {
    f1: [0.839683, 0.827724, 0.861344, 0.832244, 0.823669, 0.880519, 0.861074, 0.723443, 0.728891, 0.846048, 0.861336, 0.722772, 0.86602, 0.883156, 0.867283, 0.869309, 0.866339, 0.895622, 0.867515, 0.857725, 0.857121, 0.869482],
    fpr: [0.018763, 0.01937, 0.018045, 0.018797, 0.018875, 0.017105, 0.017963, 0.026226, 0.02559, 0.018284, 0.017789, 0.026321, 0.017834, 0.016956, 0.017619, 0.017578, 0.017634, 0.01688, 0.017527, 0.018123, 0.018016, 0.017299],
    archive: [3, 18, 17, 21],
  },
  thresholds: [0.5, 0.3, 0.55, 0.45, 0.5, 0.35, 0.15, 0.15],
  thresholdAssessment: [
    ["Default argmax", 0.9296, 0.896299, 0.017573],
    ["Adaptive thresholds", 0.928, 0.895185, 0.017565],
  ],
};
