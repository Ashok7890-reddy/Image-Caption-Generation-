"""
Enhanced BLIP Model with advanced caption generation features.
Supports beam search, temperature control, and different caption styles.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBLIPModel:
    """Enhanced BLIP model with advanced caption generation capabilities."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """Initialize the enhanced BLIP model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Style prompts for different caption styles
        self.style_prompts = {
            "descriptive": "A detailed description of",
            "funny": "A funny description of",
            "poetic": "A poetic description of",
            "formal": "A formal description of",
            "creative": "An imaginative description of",
            "technical": "A technical analysis of"
        }
        
    def generate_caption(
        self,
        image: Image.Image,
        style: str = "descriptive",
        num_captions: int = 1,
        max_length: int = 50,
        num_beams: int = 5,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> List[str]:
        """
        Generate captions for an image with various parameters.
        
        Args:
            image: PIL Image object
            style: Caption style ('descriptive', 'funny', 'poetic', 'formal', 'creative', 'technical')
            num_captions: Number of caption variations to generate
            max_length: Maximum length of generated captions
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling (higher = more creative)
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repetition
            
        Returns:
            List of generated captions
        """
        try:
            # Prepare inputs
            if style in self.style_prompts:
                prompt = self.style_prompts[style]
                inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate captions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_captions,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
            
            # Decode captions
            captions = []
            for output in outputs:
                caption = self.processor.decode(output, skip_special_tokens=True)
                # Remove the prompt if it was added
                if style in self.style_prompts and caption.startswith(self.style_prompts[style]):
                    caption = caption[len(self.style_prompts[style]):].strip()
                captions.append(caption)
            
            return captions
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return ["Error generating caption"]
    
    def batch_generate_captions(
        self,
        images: List[Image.Image],
        style: str = "descriptive",
        **kwargs
    ) -> List[List[str]]:
        """Generate captions for multiple images."""
        results = []
        for image in images:
            captions = self.generate_caption(image, style=style, **kwargs)
            results.append(captions)
        return results
    
    def get_style_options(self) -> List[str]:
        """Get available caption styles."""
        return list(self.style_prompts.keys())
    
    def fine_tune(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        save_path: str = "./fine_tuned_blip"
    ):
        """
        Fine-tune the BLIP model on a custom dataset.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            save_path: Path to save the fine-tuned model
        """
        from torch.utils.data import DataLoader
        from transformers import AdamW, get_linear_schedule_with_warmup
        from tqdm import tqdm
        
        # Set model to training mode
        self.model.train()
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        logger.info(f"Starting fine-tuning for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # Prepare inputs
                images = batch['images']
                captions = batch['captions']
                
                # Process batch
                inputs = self.processor(
                    images=images,
                    text=captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Validation if provided
            if val_dataset:
                val_loss = self._validate(val_dataset, batch_size)
                logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Save fine-tuned model
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        logger.info(f"Fine-tuned model saved to {save_path}")
    
    def _validate(self, val_dataset, batch_size: int) -> float:
        """Validate the model on validation dataset."""
        from torch.utils.data import DataLoader
        
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images']
                captions = batch['captions']
                
                inputs = self.processor(
                    images=images,
                    text=captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model."""
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.processor = BlipProcessor.from_pretrained(model_path)
        logger.info(f"Fine-tuned model loaded from {model_path}")
