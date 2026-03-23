def classify(score):
    if score < 0.3:
        return "NORMAL"
    elif score < 0.6:
        return "SUSPICIOUS"
    else:
        return "HIGHLY ANOMALOUS"