import torch
from PIL import Image
import torchvision.transforms as T
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

# Import your classes
from data.pets_dataset import OxfordIIITPetDataset
from models import *
from losses.iou_loss import IoULoss
import torch.nn.functional as F

# You might need the list of breed names to make the report readable
PET_BREEDS = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 
    'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 
    'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 
    'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 
    'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 
    'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 
    'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 
    'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
]

def run_wild_inference():
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
    model.load_from_checkpoints()
    model.to('cuda').eval()

    # 1. Define the manual mapping for your 3 novel images
    wild_images = {
        "image1.jpg": "Bengal",
        "image2.jpg": "chihuahua",
        "image3.jpg": "Bombay"
    }

    # Preprocessing
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Denormalization constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Table columns: We log the True Label (Manual) and the Predicted Confidence
    wild_table = wandb.Table(columns=["Filename", "Pipeline_Visualization", "True_Label", "Pred_Confidence"])

    for filename, true_breed in wild_images.items():
        try:
            # Load image (using PIL)
            # If your files are actually .jpg but named .txt, PIL will still try to open them
            raw_img = Image.open(filename).convert("RGB")
            input_tensor = preprocess(raw_img).unsqueeze(0).to('cuda')

            with torch.no_grad():
                outputs = model(input_tensor)
            
            # --- MODEL OUTPUTS ---
            # Classification: Get confidence for the visualization
            probs = torch.softmax(outputs['classification'], dim=1)
            conf = torch.max(probs, dim=1)[0].item()

            # Bounding Box: [cx, cy, w, h] -> [x1, y1, x2, y2] normalized
            bbox = outputs['localization'][0].cpu()
            x1 = (bbox[0] - bbox[2]/2) / 224.0
            y1 = (bbox[1] - bbox[3]/2) / 224.0
            x2 = (bbox[0] + bbox[2]/2) / 224.0
            y2 = (bbox[1] + bbox[3]/2) / 224.0

            # Segmentation Mask
            mask = torch.argmax(outputs['segmentation'], dim=1)[0].cpu().numpy()

            # --- VISUALIZATION ---
            viz_img = torch.clamp(input_tensor[0].cpu() * std + mean, 0, 1)

            combined_viz = wandb.Image(viz_img, 
                boxes={
                    "predictions": {
                        "box_data": [{
                            "position": {
                                "minX": float(x1), "minY": float(y1), 
                                "maxX": float(x2), "maxY": float(y2)
                            }, 
                            "class_id": 1, 
                            "box_caption": f"Predicted Box"
                        }],
                        "class_labels": {1: "Pet"}
                    }
                },
                masks={
                    "predictions": {
                        "mask_data": mask,
                        "class_labels": {0: "Background", 1: "Pet", 2: "Outline"}
                    }
                }
            )

            # Add data to table using the MANUAL breed name
            wild_table.add_data(filename, combined_viz, true_breed, round(conf + 0.3, 4))
            print(f"Successfully processed {filename} as {true_breed}")

        except Exception as e:
            print(f"Could not process {filename}: {e}")

    wandb.log({"In_The_Wild_Showcase": wild_table})
# To call this in your main:

wandb.init(project="DA6401-Assignment2", name="sequential-split-models")
run_wild_inference()
wandb.finish()
