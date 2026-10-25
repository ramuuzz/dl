# Image enhancement, grayscale and binary thresholding
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

# 1. Load image (place image.png beside this script)
image = Image.open("image.png").convert("RGB")
img_tensor = transforms.ToTensor()(image)

# 2. Grayscale and binary versions
gray = image.convert("L")
binary = gray.point(lambda pixel: 255 if pixel > 128 else 0)

# 3. Brightness and contrast
brightness_factor = 1.2
bright = torch.clamp(img_tensor * brightness_factor, 0, 1)
contrast_factor = 1.3
mean = bright.mean(dim=(1, 2), keepdim=True)
enhanced = torch.clamp((bright - mean) * contrast_factor + mean, 0, 1)
enhanced_image = transforms.ToPILImage()(enhanced)

# 4. Display results
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1); plt.imshow(image); plt.title("Original"); plt.axis("off")
plt.subplot(2, 2, 2); plt.imshow(enhanced_image); plt.title("Enhanced"); plt.axis("off")
plt.subplot(2, 2, 3); plt.imshow(binary, cmap="gray"); plt.title("Binary"); plt.axis("off")
plt.subplot(2, 2, 4); plt.imshow(gray, cmap="gray"); plt.title("Grayscale"); plt.axis("off")
plt.tight_layout(); plt.show()
