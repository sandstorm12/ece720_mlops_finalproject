import json
import torch
import torchvision
import shap
import numpy as np
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights


class SHAPImageExplainer:
    def __init__(self, dataset, device=None, topk=4, batch_size=50, n_evals=10000):
        self.device = device or torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(self.device).eval()
        
        self.dataset = dataset
        self.X, self.y_id, self.y_name = self.dataset._get_sample_data()
        self.num_samples = len(self.X)
        
        self.topk = topk
        self.batch_size = batch_size
        self.n_evals = n_evals

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.class_names = self._load_class_names()

        self.transform = self._build_transform()
        self.inv_transform = self._build_inv_transform()

        # prepare SHAP explainer
        example_input = self.transform(torch.Tensor(self.X[0]))
        masker_blur = shap.maskers.Image("blur(128,128)", example_input.shape)
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

    def plot_shap(self):
        # Random sample
        random_index = np.random.choice(range(self.num_samples))
        Xtr = self.transform(torch.Tensor(self.X))
        input_img = Xtr[random_index].unsqueeze(0)

        output = self.predict(input_img)
        predicted_class = torch.argmax(output, axis=1).cpu().numpy()
        true_label = [self.y_name[random_index]]
        print(f"Predicted: {np.array(self.class_names)[predicted_class]}, True: {true_label}")

        # Explain with SHAP
        shap_values = self.explainer(
            input_img,
            max_evals=self.n_evals,
            batch_size=self.batch_size,
            outputs=shap.Explanation.argsort.flip[:self.topk],
        )

        shap_values.data = self.inv_transform(shap_values.data).cpu().numpy()[0]
        shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

        shap.image_plot(
            shap_values=shap_values.values,
            pixel_values=shap_values.data,
            labels=shap_values.output_names,
            true_labels=true_label
        )
