from ultralytics import YOLO
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import warnings
warnings.filterwarnings("ignore")

# Load a model
model = YOLO('./yolov8m-pose.pt')  # load a custom model

def get_points(path):
  img = cv2.imread(path)
  img = cv2.resize(img, (640, 640))
  results = model(img, verbose = False)  # predict on an image
  keypoints = results[0].keypoints  # Keypoints object for pose outputs
  result = keypoints.xy.detach().cpu().numpy()[0].reshape(-1).tolist()
  return result

if __name__ == "__main__":
  # Initialize the Parser
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", type = str, required=True)
  args = parser.parse_args()


  df = pd.read_csv("./data.csv", delimiter = " ")
  df.dropna(inplace = True)
  target = df.pop("labels")
  data = df

  norm = StandardScaler()
  data = norm.fit_transform(data)

  svm = joblib.load("./model.pkl")
  a = get_points(args.path)
  classes = ["pullshot", "legglance-flick", "sweep", "drive"]

  print(classes[svm.predict(norm.transform([a]))[0]])

