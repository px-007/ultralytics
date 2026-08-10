from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-depth.pt")  # load an official model

# Predict with the model
results = model(r"D:\BaiduNetdiskDownload\datasets\qiping\qiping_1012.jpg")  # predict on an image

# Access the results
for result in results:
    depth_map = result.depth.data.cpu().numpy()  # torch.Tensor -> NumPy float32, shape (H, W), meters