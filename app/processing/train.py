from ultralytics import YOLO

model1=YOLO('/content/yolo11n.pt')
results = model1.train(
    data='/content/drive/MyDrive/XLA/rgb_dataset/data.yaml',
    epochs=100,
    imgsz=960,  
    batch=32,   
    name='train_enhanced_LAB',
    project='/content/drive/MyDrive/XLA/model_train/clahe_dataset_result'
)