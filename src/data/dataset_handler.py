"""
Dataset handling and preprocessing for image captioning.
Supports Flickr8k and custom datasets.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
import zipfile
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCaptionDataset(Dataset):
    """Custom dataset for image captioning."""
    
    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        processor,
        max_length: int = 128
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to images
            captions: List of captions corresponding to images
            processor: BLIP processor for tokenization
            max_length: Maximum length for captions
        """
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
        self.max_length = max_length
        
        assert len(image_paths) == len(captions), "Number of images and captions must match"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            # Load and process image
            image = Image.open(self.image_paths[idx]).convert('RGB')
            caption = self.captions[idx]
            
            # Process with BLIP processor
            encoding = self.processor(
                images=image,
                text=caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                'images': image,
                'captions': caption,
                'pixel_values': encoding['pixel_values'].squeeze(),
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
        except Exception as e:
            logger.error(f"Error loading item {idx}: {str(e)}")
            # Return a dummy item in case of error
            dummy_image = Image.new('RGB', (224, 224), color='black')
            dummy_caption = "dummy caption"
            encoding = self.processor(
                images=dummy_image,
                text=dummy_caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {
                'images': dummy_image,
                'captions': dummy_caption,
                'pixel_values': encoding['pixel_values'].squeeze(),
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }

class Flickr8kDatasetHandler:
    """Handler for Flickr8k dataset."""
    
    def __init__(self, data_dir: str = "./data/flickr8k"):
        """Initialize the Flickr8k dataset handler."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.data_dir / "Images"
        self.captions_file = self.data_dir / "captions.txt"
        
    def download_dataset(self):
        """Download and extract Flickr8k dataset."""
        logger.info("Downloading Flickr8k dataset...")
        
        # Note: In a real implementation, you would need to get the dataset from Kaggle
        # For this demo, we'll create a sample dataset structure
        self._create_sample_dataset()
        
    def _create_sample_dataset(self):
        """Create a sample dataset for demonstration."""
        logger.info("Creating sample dataset for demonstration...")
        
        # Create sample images directory
        self.images_dir.mkdir(exist_ok=True)
        
        # Create sample captions file
        sample_captions = [
            "sample_1.jpg,A dog running in the park",
            "sample_1.jpg,A happy dog playing outside",
            "sample_2.jpg,A cat sitting on a windowsill",
            "sample_2.jpg,A white cat looking out the window",
            "sample_3.jpg,Children playing in a playground",
            "sample_3.jpg,Kids having fun on swings and slides"
        ]
        
        with open(self.captions_file, 'w') as f:
            f.write("image,caption\n")
            for caption in sample_captions:
                f.write(caption + "\n")
        
        # Create sample images (placeholder)
        from PIL import Image, ImageDraw, ImageFont
        
        sample_images = [
            ("sample_1.jpg", "Dog in Park", (255, 200, 100)),
            ("sample_2.jpg", "Cat on Window", (200, 255, 200)),
            ("sample_3.jpg", "Children Playing", (200, 200, 255))
        ]
        
        for filename, text, color in sample_images:
            img = Image.new('RGB', (224, 224), color=color)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
                draw.text((10, 100), text, fill=(0, 0, 0), font=font)
            except:
                draw.text((10, 100), text, fill=(0, 0, 0))
            img.save(self.images_dir / filename)
        
        logger.info("Sample dataset created successfully!")
    
    def load_dataset(self, processor, train_split: float = 0.8) -> Tuple[ImageCaptionDataset, ImageCaptionDataset]:
        """
        Load and split the dataset into train and validation sets.
        
        Args:
            processor: BLIP processor
            train_split: Fraction of data to use for training
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if not self.captions_file.exists():
            logger.info("Dataset not found. Creating sample dataset...")
            self._create_sample_dataset()
        
        # Load captions
        df = pd.read_csv(self.captions_file)
        
        # Prepare image paths and captions
        image_paths = []
        captions = []
        
        for _, row in df.iterrows():
            image_path = self.images_dir / row['image']
            if image_path.exists():
                image_paths.append(str(image_path))
                captions.append(row['caption'])
        
        logger.info(f"Loaded {len(image_paths)} image-caption pairs")
        
        # Split dataset
        split_idx = int(len(image_paths) * train_split)
        
        train_images = image_paths[:split_idx]
        train_captions = captions[:split_idx]
        val_images = image_paths[split_idx:]
        val_captions = captions[split_idx:]
        
        # Create datasets
        train_dataset = ImageCaptionDataset(train_images, train_captions, processor)
        val_dataset = ImageCaptionDataset(val_images, val_captions, processor) if val_images else None
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        if val_dataset:
            logger.info(f"Validation dataset: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def get_sample_images(self, num_samples: int = 5) -> List[Tuple[str, str]]:
        """Get sample images and their captions for demonstration."""
        if not self.captions_file.exists():
            self._create_sample_dataset()
        
        df = pd.read_csv(self.captions_file)
        samples = []
        
        for _, row in df.head(num_samples).iterrows():
            image_path = self.images_dir / row['image']
            if image_path.exists():
                samples.append((str(image_path), row['caption']))
        
        return samples

class CustomDatasetHandler:
    """Handler for custom datasets."""
    
    def __init__(self, data_dir: str):
        """Initialize custom dataset handler."""
        self.data_dir = Path(data_dir)
    
    def create_dataset_from_folder(
        self,
        images_folder: str,
        captions_file: str,
        processor
    ) -> ImageCaptionDataset:
        """
        Create dataset from a folder of images and captions file.
        
        Args:
            images_folder: Path to folder containing images
            captions_file: Path to CSV file with image names and captions
            processor: BLIP processor
            
        Returns:
            ImageCaptionDataset
        """
        df = pd.read_csv(captions_file)
        
        image_paths = []
        captions = []
        
        for _, row in df.iterrows():
            image_path = Path(images_folder) / row['image']
            if image_path.exists():
                image_paths.append(str(image_path))
                captions.append(row['caption'])
        
        return ImageCaptionDataset(image_paths, captions, processor)
    
    def export_captions_to_csv(
        self,
        image_paths: List[str],
        captions: List[str],
        output_file: str
    ):
        """Export generated captions to CSV file."""
        df = pd.DataFrame({
            'image_path': image_paths,
            'caption': captions
        })
        df.to_csv(output_file, index=False)
        logger.info(f"Captions exported to {output_file}")
