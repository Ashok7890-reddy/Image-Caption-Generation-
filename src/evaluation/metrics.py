"""
Evaluation metrics for image captioning models.
Implements BLEU, CIDEr, METEOR, and ROUGE-L scores.
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
from typing import List, Dict, Tuple
import logging
import json
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptionEvaluator:
    """Evaluator for image captioning models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.smoothing = SmoothingFunction().method4
        
    def compute_bleu_scores(
        self,
        references: List[List[str]],
        candidates: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4).
        
        Args:
            references: List of reference captions for each image
            candidates: List of generated captions
            
        Returns:
            Dictionary with BLEU scores
        """
        bleu_scores = {
            'BLEU-1': [],
            'BLEU-2': [],
            'BLEU-3': [],
            'BLEU-4': []
        }
        
        for ref_list, candidate in zip(references, candidates):
            # Tokenize
            ref_tokens = [ref.lower().split() for ref in ref_list]
            candidate_tokens = candidate.lower().split()
            
            # Compute BLEU scores
            bleu_scores['BLEU-1'].append(
                sentence_bleu(ref_tokens, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
            )
            bleu_scores['BLEU-2'].append(
                sentence_bleu(ref_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
            )
            bleu_scores['BLEU-3'].append(
                sentence_bleu(ref_tokens, candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
            )
            bleu_scores['BLEU-4'].append(
                sentence_bleu(ref_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
            )
        
        # Average scores
        return {key: np.mean(scores) for key, scores in bleu_scores.items()}
    
    def compute_meteor_score(
        self,
        references: List[List[str]],
        candidates: List[str]
    ) -> float:
        """
        Compute METEOR score.
        
        Args:
            references: List of reference captions for each image
            candidates: List of generated captions
            
        Returns:
            Average METEOR score
        """
        meteor_scores = []
        
        for ref_list, candidate in zip(references, candidates):
            # METEOR expects a single reference string
            # We'll use the first reference or concatenate all
            reference = ref_list[0] if ref_list else ""
            
            try:
                score = meteor_score([reference.lower().split()], candidate.lower().split())
                meteor_scores.append(score)
            except Exception as e:
                logger.warning(f"Error computing METEOR score: {e}")
                meteor_scores.append(0.0)
        
        return np.mean(meteor_scores)
    
    def compute_rouge_l(
        self,
        references: List[List[str]],
        candidates: List[str]
    ) -> float:
        """
        Compute ROUGE-L score.
        
        Args:
            references: List of reference captions for each image
            candidates: List of generated captions
            
        Returns:
            Average ROUGE-L score
        """
        rouge_scores = []
        
        for ref_list, candidate in zip(references, candidates):
            max_rouge = 0
            candidate_tokens = candidate.lower().split()
            
            for reference in ref_list:
                ref_tokens = reference.lower().split()
                lcs_length = self._lcs_length(ref_tokens, candidate_tokens)
                
                if len(ref_tokens) == 0 or len(candidate_tokens) == 0:
                    rouge = 0
                else:
                    precision = lcs_length / len(candidate_tokens)
                    recall = lcs_length / len(ref_tokens)
                    
                    if precision + recall == 0:
                        rouge = 0
                    else:
                        rouge = 2 * precision * recall / (precision + recall)
                
                max_rouge = max(max_rouge, rouge)
            
            rouge_scores.append(max_rouge)
        
        return np.mean(rouge_scores)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute the length of the longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def compute_cider_score(
        self,
        references: List[List[str]],
        candidates: List[str]
    ) -> float:
        """
        Simplified CIDEr score implementation.
        Note: This is a simplified version. For full CIDEr, use pycocoevalcap.
        
        Args:
            references: List of reference captions for each image
            candidates: List of generated captions
            
        Returns:
            Simplified CIDEr score
        """
        try:
            from collections import Counter
            import math
            
            cider_scores = []
            
            # Build document frequency for all references
            all_refs = []
            for ref_list in references:
                all_refs.extend(ref_list)
            
            # Compute n-gram document frequencies
            doc_freq = {}
            for ref in all_refs:
                tokens = ref.lower().split()
                ngrams = set()
                for n in range(1, 5):  # 1-4 grams
                    for i in range(len(tokens) - n + 1):
                        ngram = ' '.join(tokens[i:i+n])
                        ngrams.add(ngram)
                
                for ngram in ngrams:
                    doc_freq[ngram] = doc_freq.get(ngram, 0) + 1
            
            total_docs = len(all_refs)
            
            for ref_list, candidate in zip(references, candidates):
                candidate_tokens = candidate.lower().split()
                
                # Get candidate n-grams
                candidate_ngrams = {}
                for n in range(1, 5):
                    for i in range(len(candidate_tokens) - n + 1):
                        ngram = ' '.join(candidate_tokens[i:i+n])
                        candidate_ngrams[ngram] = candidate_ngrams.get(ngram, 0) + 1
                
                # Compute CIDEr for this example
                cider_n_scores = []
                
                for n in range(1, 5):
                    # Get reference n-grams
                    ref_ngrams = {}
                    for ref in ref_list:
                        ref_tokens = ref.lower().split()
                        for i in range(len(ref_tokens) - n + 1):
                            ngram = ' '.join(ref_tokens[i:i+n])
                            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
                    
                    # Compute similarity
                    similarity = 0
                    for ngram in candidate_ngrams:
                        if ngram in ref_ngrams:
                            tf_idf_candidate = candidate_ngrams[ngram] * math.log(total_docs / max(1, doc_freq.get(ngram, 1)))
                            tf_idf_ref = ref_ngrams[ngram] * math.log(total_docs / max(1, doc_freq.get(ngram, 1)))
                            similarity += tf_idf_candidate * tf_idf_ref
                    
                    # Normalize
                    candidate_norm = sum([(candidate_ngrams[ng] * math.log(total_docs / max(1, doc_freq.get(ng, 1))))**2 for ng in candidate_ngrams])**0.5
                    ref_norm = sum([(ref_ngrams[ng] * math.log(total_docs / max(1, doc_freq.get(ng, 1))))**2 for ng in ref_ngrams])**0.5
                    
                    if candidate_norm > 0 and ref_norm > 0:
                        cider_n_scores.append(similarity / (candidate_norm * ref_norm))
                    else:
                        cider_n_scores.append(0)
                
                cider_scores.append(np.mean(cider_n_scores))
            
            return np.mean(cider_scores)
            
        except Exception as e:
            logger.warning(f"Error computing CIDEr score: {e}")
            return 0.0
    
    def evaluate_model(
        self,
        references: List[List[str]],
        candidates: List[str],
        save_results: bool = True,
        results_file: str = "evaluation_results.json"
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of the model.
        
        Args:
            references: List of reference captions for each image
            candidates: List of generated captions
            save_results: Whether to save results to file
            results_file: File to save results
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Computing evaluation metrics...")
        
        # Compute all metrics
        bleu_scores = self.compute_bleu_scores(references, candidates)
        meteor = self.compute_meteor_score(references, candidates)
        rouge_l = self.compute_rouge_l(references, candidates)
        cider = self.compute_cider_score(references, candidates)
        
        results = {
            **bleu_scores,
            'METEOR': meteor,
            'ROUGE-L': rouge_l,
            'CIDEr': cider
        }
        
        # Print results
        logger.info("Evaluation Results:")
        for metric, score in results.items():
            logger.info(f"{metric}: {score:.4f}")
        
        # Save results
        if save_results:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
        
        return results
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Compare multiple models and generate a comparison report.
        
        Args:
            model_results: Dictionary mapping model names to their results
            
        Returns:
            Formatted comparison report
        """
        if not model_results:
            return "No model results to compare."
        
        metrics = list(next(iter(model_results.values())).keys())
        
        report = "Model Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        # Create comparison table
        report += f"{'Metric':<12}"
        for model_name in model_results.keys():
            report += f"{model_name:<12}"
        report += "\n" + "-" * (12 + 12 * len(model_results)) + "\n"
        
        for metric in metrics:
            report += f"{metric:<12}"
            for model_name, results in model_results.items():
                score = results.get(metric, 0.0)
                report += f"{score:<12.4f}"
            report += "\n"
        
        # Find best model for each metric
        report += "\nBest Performance:\n"
        for metric in metrics:
            best_model = max(model_results.keys(), key=lambda m: model_results[m].get(metric, 0))
            best_score = model_results[best_model][metric]
            report += f"{metric}: {best_model} ({best_score:.4f})\n"
        
        return report
