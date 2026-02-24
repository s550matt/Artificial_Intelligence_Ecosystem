from PIL import Image
import matplotlib.pyplot as plt
import os

def apply_bw_filter(image_path, output_path="bw_image.png"):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))

        # Convert image to black and white (grayscale)
        img_bw = img_resized.convert("L")

        plt.imshow(img_bw, cmap='gray')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Processed image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("Black & White Image Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        # derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_bw{ext}"

        apply_bw_filter(image_path, output_file)