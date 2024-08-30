import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def process_image(image_path, mask_generator, output_dir):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, mask in enumerate(masks):
        show_mask(mask['segmentation'], plt.gca(), True)
        plt.title(f"Mask {i + 1}, Score: {mask['stability_score']:.3f}", fontsize=18)
    plt.axis('off')

    output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.png', '.jpg'))
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sam_checkpoint = "D:/SAM/checkpoint/sam_vit_h.pth"
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, mask_generator, output_dir)
            print(f"Processed and saved {filename}")

if __name__ == "__main__":
    input_dir = "C:/Users/11385/Desktop/LLimages4"  # 输入图像文件夹路径
    output_dir = "C:/Users/11385/Desktop/LLimages4_output"  # 输出结果文件夹路径
    main(input_dir, output_dir)
