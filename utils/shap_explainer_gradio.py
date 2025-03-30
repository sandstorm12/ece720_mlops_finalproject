import json
import torch
import torchvision
import shap
import numpy as np
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from io import BytesIO


class SHAPImageExplainer:
    def __init__(self, device=None, topk=1, batch_size=50, n_evals=10000):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(self.device).eval()
        
        self.topk = topk
        self.batch_size = batch_size
        self.n_evals = n_evals

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.class_names = self._load_class_names()

        self.transform = self._build_transform()
        self.inv_transform = self._build_inv_transform()
        
        masker_blur = shap.maskers.Image("blur(128,128)", (224, 224, 3))
        self.explainer = shap.Explainer(self.predict, masker_blur, output_names=self.class_names)

    def _load_class_names(self):
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with open(shap.datasets.cache(url)) as file:
            return [v[1] for v in json.load(file).values()]

    def _build_transform(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(self.nhwc_to_nchw),
            torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
            torchvision.transforms.Normalize(mean=self.mean, std=self.std),
            torchvision.transforms.Lambda(self.nchw_to_nhwc),
        ])

    def _build_inv_transform(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(self.nhwc_to_nchw),
            torchvision.transforms.Normalize(
                mean=(-1 * np.array(self.mean) / np.array(self.std)).tolist(),
                std=(1 / np.array(self.std)).tolist(),
            ),
            torchvision.transforms.Lambda(self.nchw_to_nhwc),
        ])

    def nhwc_to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
        return x

    def nchw_to_nhwc(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
        return x

    def predict(self, img: np.ndarray) -> torch.Tensor:
        img = self.nhwc_to_nchw(torch.Tensor(img))
        img = img.to(self.device)
        with torch.no_grad():
            output = self.model(img)
        return output
    
    def explain(self, img):
        Xtr = self.transform(torch.Tensor(img))
        input_img = Xtr.unsqueeze(0)

        output = self.predict(input_img)
        predicted_class = torch.argmax(output, axis=1).cpu().numpy()

        # Explain with SHAP
        shap_values = self.explainer(
            input_img,
            max_evals=self.n_evals,
            batch_size=self.batch_size,
            outputs=shap.Explanation.argsort.flip[:self.topk],
        )

        shap_values.data = self.inv_transform(shap_values.data).cpu().numpy()[0]
        shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

        plt.figure()
        shap.image_plot(
            shap_values=shap_values.values,
            pixel_values=shap_values.data,
            labels=shap_values.output_names,
            true_labels=None,
            show=False  # Prevent displaying the plot
        )

        # Convert plot to NumPy array
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        buf.seek(0)
        
        # Convert buffer to NumPy array
        from PIL import Image
        img = Image.open(buf)
        numpy_image = np.array(img)
        buf.close()

        return numpy_image, np.array(shap_values.values[0]), predicted_class
