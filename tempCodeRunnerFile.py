
model = YOLO(os.path.join(ROOT_DIR, f"results/trainM{ranEpochs}epochs/weights/best.pt"))
model_RF = joblib.load(os.path.join(ROOT_DIR, "model.joblib"))
class_labels = ["Green", "Red", "Yellow", "Off"]