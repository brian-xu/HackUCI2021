import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from main import SimpleCNN


class Predictor:
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __init__(self, gpu: bool):
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and gpu) else 'cpu')
        self.net = SimpleCNN()
        self.net.load_state_dict(torch.load('./models/model.pt', map_location=self.device))
        self.net.eval()

    def predict(self, img: np.array) -> [(np.array, torch.Tensor)]:
        with torch.no_grad():
            pred = []
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            model_input = Predictor.transform(img)
            model_input = model_input.unsqueeze(0)
            output = self.net(model_input).squeeze(1)
            pred.append((np.asarray(img), torch.argmax(output.cpu())))

        return pred
