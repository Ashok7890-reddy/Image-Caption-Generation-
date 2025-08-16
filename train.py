"""
Training script for fine-tuning the BLIP model on custom datasets.
"""

import torch
import argparse
import logging
from pathlib import Path

from src.models.enhanced_blip import EnhancedBLIPModel
from src.data.dataset_handler import Flickr8kDatasetHandler
from src.evaluation.metrics import CaptionEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune BLIP model")
    parser.add_argument("--data_dir", type=str, default="./data/flickr8k", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./models/fine_tuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    
    args = parser.parse_args()
    
    # Initialize components
    logger.info("Initializing model and dataset handler...")
    model = EnhancedBLIPModel()
    dataset_handler = Flickr8kDatasetHandler(args.data_dir)
    
    # Load dataset
    logger.info("Loading dataset...")
    train_dataset, val_dataset = dataset_handler.load_dataset(model.processor)
    
    if len(train_dataset) == 0:
        logger.error("No training data found. Please check your dataset.")
        return
    
    # Fine-tune model
    logger.info("Starting fine-tuning...")
    model.fine_tune(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_path=args.output_dir
    )
    
    # Evaluate if requested
    if args.evaluate and val_dataset:
        logger.info("Running evaluation...")
        evaluator = CaptionEvaluator()
        
        # Generate captions for validation set
        references = []
        candidates = []
        
        for i in range(min(10, len(val_dataset))):  # Evaluate on first 10 samples
            sample = val_dataset[i]
            image = sample['images']
            reference = sample['captions']
            
            generated = model.generate_caption(image, num_captions=1)
            
            references.append([reference])
            candidates.append(generated[0])
        
        # Compute metrics
        results = evaluator.evaluate_model(
            references=references,
            candidates=candidates,
            save_results=True,
            results_file=f"{args.output_dir}/evaluation_results.json"
        )
        
        logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()
