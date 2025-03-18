import os
import shutil

with open('dataset/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'dataset/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'dataset/tiny-imagenet-200/val/images/{fn}', f'dataset/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('dataset/tiny-imagenet-200/val/images')

from torchvision.datasets import ImageFolder
import torchvision.transforms as T

'''
1. T.Resize((224, 224))
	•	Resizes the image to 224x224 pixels.
	•	Many pretrained models (like ResNet, VGG) expect inputs of this size.

2. T.ToTensor()
	•	Converts the PIL image (or NumPy array) to a PyTorch tensor.
	•	Changes the shape from (H, W, C) to (C, H, W) (PyTorch expects channel-first format).
	•	Converts pixel values from [0, 255] (uint8) to [0, 1] (float32).

3. T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	•	Normalizes each color channel using the given mean and standard deviation values.
	•	These values are from ImageNet dataset statistics (which most pretrained models were trained on).
	•	Formula applied per channel:
'''
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# root/{class}/x001.jpg

tiny_imagenet_dataset_train = ImageFolder(root='dataset/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='dataset/tiny-imagenet-200/val', transform=transform)

print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")