import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

def get_transforms(mode='train', img_size=(384, 384)):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

def parse_xml_annotation(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            objects.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
        return objects
    except ET.ParseError:
        print(f"Warning: Invalid XML file {xml_path}, skipping annotations")
        return []

class TBXRayDataset(Dataset):
    def __init__(self, data_dicts, transform=None):
        self.data_dicts = data_dicts
        self.transform = transform

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data = self.data_dicts[idx]
        try:
            image = Image.open(data['image']).convert('L')
        except Exception as e:
            print(f"Warning: Failed to load image {data['image']}: {e}")
            # Return a dummy image and label to avoid crashing
            image = Image.new('L', (384, 384), color=0)
            return {
                'image': self.transform(image) if self.transform else image,
                'label': torch.tensor(data['label'], dtype=torch.long)
            }
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'label': torch.tensor(data['label'], dtype=torch.long)
        }

def get_data_dicts_with_annotations(data_dir):
    class_names = ['Healthy', 'Sick', 'TB']
    data_dicts = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, 'imgs', class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping class {class_name}")
            continue
        image_paths = glob.glob(os.path.join(class_dir, '*.png')) + \
                      glob.glob(os.path.join(class_dir, '*.jpg'))
        if not image_paths:
            print(f"Warning: No images found in {class_dir} for class {class_name}")
            continue
        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(data_dir, 'annotations', 'xml', base + '.xml')
            annotations = parse_xml_annotation(xml_path) if os.path.exists(xml_path) else []
            data_dicts.append({
                'image': img_path,
                'label': label_idx,
                'annotations': annotations
            })
    
    # Debug: Print class distribution
    if data_dicts:
        labels = [d['label'] for d in data_dicts]
        class_counts = np.bincount(labels, minlength=3)
        print(f"Full dataset class distribution: {class_counts} (Healthy: {class_counts[0]}, Sick: {class_counts[1]}, TB: {class_counts[2]})")
        if class_counts[0] == 0:
            print("Error: No samples found for Healthy class. Check dataset directory structure or add Healthy images.")
    else:
        print("Error: No valid data found in dataset. Check directory structure and image files.")
    
    return data_dicts

def get_dataloader_with_annotations(data_dir, batch_size, mode='train', img_size=(384, 384), val_split=0.2):
    all_data = get_data_dicts_with_annotations(data_dir)
    if not all_data:
        raise ValueError("No data available after loading. Check dataset and directory structure.")
    
    train_data, val_data = train_test_split(
        all_data,
        test_size=val_split,
        stratify=[d['label'] for d in all_data],
        random_state=42
    )
    
    dataset = TBXRayDataset(
        train_data if mode == 'train' else val_data,
        transform=get_transforms(mode, img_size)
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=4,
        pin_memory=True
    )
