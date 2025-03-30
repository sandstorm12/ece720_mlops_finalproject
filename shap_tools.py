import cv2

from utils.shap_explainer_gradio import SHAPImageExplainer


def _build_explainer():
    explainer = SHAPImageExplainer()

    return explainer


if __name__ == "__main__":
    explainer = _build_explainer()

    img_path = "/home/hamid/Documents/uofa/MLOPs/imagenet_val_s/ImageNetS50/validation/n01443537/ILSVRC2012_val_00004677.JPEG"
    img_sample = cv2.imread(img_path)

    image_rgb = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, (256, 256))
    h, w, _ = image_resized.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    image_cropped = image_resized[top:top+224, left:left+224]

    explaination_img = explainer.explain(image_cropped)
    explaination_img = cv2.cvtColor(explaination_img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Explaination", explaination_img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
