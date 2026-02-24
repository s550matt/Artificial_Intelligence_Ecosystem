import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pretrained model
model = MobileNetV2(weights="imagenet")

# For MobileNetV2, this is a good "last conv" layer for Grad-CAM
LAST_CONV_LAYER_NAME = "Conv_1"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Creates a Grad-CAM heatmap for a given preprocessed image batch (shape: (1, 224, 224, 3)).
    """
    # Build a model that maps input image -> activations of last conv layer + model predictions
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)  # conv_outputs: (1, h, w, channels)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]  # score for chosen class

    # Gradient of the class score with respect to the feature map activations
    grads = tape.gradient(class_channel, conv_outputs)

    # Global-average-pool the gradients over the spatial dimensions (h, w)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (channels,)

    # Weight the channels by their importance
    conv_outputs = conv_outputs[0]  # shape: (h, w, channels)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # shape: (h, w)

    # Apply ReLU and normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / max_val, heatmap)

    return heatmap.numpy()


def overlay_heatmap_on_image(original_img_array, heatmap, alpha=0.4):
    """
    Overlays a heatmap (h, w) on an RGB image array (224, 224, 3).
    Returns an RGB uint8 image (224, 224, 3).
    """
    # Convert heatmap to RGB using a simple red overlay (no matplotlib/cv2)
    heatmap_rgb = np.zeros_like(original_img_array, dtype=np.float32)
    heatmap_rgb[..., 0] = heatmap  # put heat in red channel

    # Normalize original to [0,1] float for blending
    img_float = original_img_array.astype(np.float32) / 255.0

    # Blend: original + alpha*heatmap
    overlay = img_float * (1 - alpha) + heatmap_rgb * alpha
    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)


def classify_image(image_path):
    try:
        # Load image twice:
        # - original_img: for saving/displaying overlay
        # - img_array: for model prediction (preprocessed)
        original_img = image.load_img(image_path, target_size=(224, 224))
        original_arr = image.img_to_array(original_img).astype(np.uint8)

        img_array = original_arr.astype(np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("\nTop-3 Predictions for", image_path)
        for i, (_, label, score) in enumerate(decoded_predictions):
            print(f"  {i + 1}: {label} ({score:.2f})")

        # Grad-CAM for top-1 prediction
        top1_index = int(np.argmax(predictions[0]))
        heatmap = make_gradcam_heatmap(
            img_array=img_array,
            model=model,
            last_conv_layer_name=LAST_CONV_LAYER_NAME,
            pred_index=top1_index
        )

        # Resize heatmap to 224x224 (MobileNetV2 last conv output is smaller)
        heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (224, 224)).numpy().squeeze()

        overlay = overlay_heatmap_on_image(original_arr, heatmap_resized, alpha=0.45)

        # Save overlay image next to input file name
        out_path = image_path.rsplit(".", 1)[0] + "_gradcam.png"
        image.save_img(out_path, overlay)
        print(f"Saved Grad-CAM overlay to: {out_path}")

    except Exception as e:
        print(f"Error processing '{image_path}': {e}")


if __name__ == "__main__":
    print("Image Classifier + Grad-CAM (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename: ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        classify_image(image_path)