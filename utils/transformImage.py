import os
import shutil

with open('dataset/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'dataset/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'dataset/tiny-imagenet-200/val/images/{fn}', f'dataset/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('dataset/tiny-imagenet-200/val/images')