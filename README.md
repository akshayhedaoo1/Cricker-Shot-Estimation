## Detecting Cricket Shot using Body Points 

1. We have using yolov8 pose estimation which having 17 body points. 
2. These Body points are labelled with their Shot class and collected inside a CSV File.
3. Use a SVM to classify those points into four categories.

For more information, Please refer train.ipynb notebook.
This is just a sample demonstration. For Good Results, we need Quality Dataset and create Good Feature Engineering.

### For Inference

python predict.py -i ./legflick.jpg

