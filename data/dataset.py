from PIL import Image
import json
import os
import paddle.io as data
import paddle
identity = lambda x: x


class SimpleDataset(data.Dataset):
    def __init__(self, data_file, transform, target_transform=identity):
        super().__init__()
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = paddle.to_tensor(self.transform(img))
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])



