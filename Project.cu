import torch

torch.cuda.set_device(0) # Set to your desired GPU number

model = YOLO("yolov8n.yaml", device='gpu')

