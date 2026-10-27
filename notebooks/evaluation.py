"""
Part 3: Comprehensive Evaluation Module
This module implements rigorous evaluation metrics for the Agentic RAG system.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import re


class RetrievalEvaluator:
    """
    Evaluates retrieval accuracy using standard IR metrics.
    """
    
    @staticmethod
    def extract_article_ids(text: str) -> set:
        """Extract article IDs from text."""
        # Pattern to match article IDs
        ids = re.findall(r'Article ID:\s*(\d+)', text)
        ids += re.findall(r'article_id[\'\"]\s*:\s*[\'\"]*(\d+)', text)
        return set(ids)
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        Calculate Precision@K.
        
        Precision@K = (# relevant retrieved in top K) / K
        """
        if not retrieved:
            return 0.0
            
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        relevant_retrieved = retrieved_set.intersection(relevant_set)
        return len(relevant_retrieved) / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        Calculate Recall@K.
        
        Recall@K = (# relevant retrieved in top K) / (total # relevant)
        """
        if not relevant:
            return 0.0
            
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        relevant_retrieved = retrieved_set.intersection(relevant_set)
        return len(relevant_retrieved) / len(relevant_set)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / (rank of first relevant document)
        """
        relevant_set = set(relevant)
        
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        """
        relevant_set = set(relevant)
        
        # Calculate DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


class ResponseEvaluator:
    """
    Evaluates response quality using text similarity metrics.
    """
    
    def __init__(self, rouge_scorer_obj, bleu_smoother, sentence_model):
        """Initialize with required evaluation tools."""
        self.rouge_scorer = rouge_scorer_obj
        self.bleu_smoother = bleu_smoother
        self.sentence_model = sentence_model
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """
        Calculate BLEU score.
        
        Returns:
            BLEU score (0-1)
        """
        try:
            # Tokenize
            reference_tokens = reference.lower().split()
            prediction_tokens = prediction.lower().split()
            
            # Import here to avoid issues
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Calculate BLEU with smoothing - use method7 for better compatibility
            smoother = SmoothingFunction()
            score = sentence_bleu(
                [reference_tokens],
                prediction_tokens,
                smoothing_function=smoother.method7
            )
            
            return float(score)
        except Exception as e:
            return 0.0
    
    def calculate_semantic_similarity(self, prediction: str, reference: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            from sentence_transformers import util
            
            # Generate embeddings
            pred_embedding = self.sentence_model.encode(prediction, convert_to_tensor=True)
            ref_embedding = self.sentence_model.encode(reference, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(pred_embedding, ref_embedding)
            
            return float(similarity.item())
        except Exception as e:
            return 0.0


class LLMJudge:
    """
    Implements LLM-as-Judge evaluation methodology.
    """
    
    def __init__(self, llm):
        """Initialize with language model."""
        self.llm = llm
    
    def evaluate_response(self, question: str, prediction: str, reference: str) -> Dict[str, Any]:
        """
        Evaluate response quality using LLM as judge.
        
        Returns:
            Dictionary with relevance, accuracy, completeness scores and explanation
        """
        judge_prompt = f"""You are an expert evaluator assessing AI-generated responses about CleanTech topics.

Question: {question}

Generated Answer: {prediction}

Reference Answer: {reference}

Please evaluate the generated answer on the following criteria (rate 1-5):

1. RELEVANCE: Does it directly address the question?
2. ACCURACY: Is the information factually correct compared to the reference?
3. COMPLETENESS: Does it cover the key points from the reference?
4. CLARITY: Is it well-written and easy to understand?

Provide your evaluation in the following format:
RELEVANCE: [score 1-5]
ACCURACY: [score 1-5]
COMPLETENESS: [score 1-5]
CLARITY: [score 1-5]
OVERALL: [average score]
EXPLANATION: [brief explanation of your scoring]"""

        try:
            response = self.llm.invoke(judge_prompt)
            content = response.content
            
            # Parse scores
            relevance = self._extract_score(content, "RELEVANCE")
            accuracy = self._extract_score(content, "ACCURACY")
            completeness = self._extract_score(content, "COMPLETENESS")
            clarity = self._extract_score(content, "CLARITY")
            
            # Calculate overall
            overall = (relevance + accuracy + completeness + clarity) / 4.0
            
            # Extract explanation
            explanation = self._extract_explanation(content)
            
            return {
                'relevance': relevance,
                'accuracy': accuracy,
                'completeness': completeness,
                'clarity': clarity,
                'overall': overall,
                'explanation': explanation
            }
        
        except Exception as e:
            return {
                'relevance': 0,
                'accuracy': 0,
                'completeness': 0,
                'clarity': 0,
                'overall': 0,
                'explanation': f"Error: {str(e)}"
            }
    
    @staticmethod
    def _extract_score(text: str, metric: str) -> float:
        """Extract numerical score from LLM response."""
        pattern = f"{metric}:\\s*([0-5](?:\\.[0-9]+)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.0
    
    @staticmethod
    def _extract_explanation(text: str) -> str:
        """Extract explanation from LLM response."""
        pattern = r"EXPLANATION:\s*(.+?)(?:\n\n|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No explanation provided"


def evaluate_system_comprehensive(
    agent,
    test_data: pd.DataFrame,
    retrieval_evaluator: RetrievalEvaluator,
    response_evaluator: ResponseEvaluator,
    llm_judge: LLMJudge,
    max_queries: int = None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on the system.
    
    Args:
        agent: The agentic system to evaluate
        test_data: DataFrame with test queries
        retrieval_evaluator: Retrieval metrics evaluator
        response_evaluator: Response quality evaluator
        llm_judge: LLM-based evaluator
        max_queries: Maximum number of queries to evaluate (None for all)
    
    Returns:
        Dictionary containing all evaluation metrics and results
    """
    results = []
    
    # Limit number of queries if specified
    if max_queries:
        test_data = test_data.head(max_queries)
    
    print(f"Evaluating {len(test_data)} queries...")
    
    for idx, row in test_data.iterrows():
        print(f"\nEvaluating query {idx + 1}/{len(test_data)}")
        
        query = row['question']
        reference_answer = row.get('answer', '')
        
        # Initialize result dict with ALL required keys and default values
        result_dict = {
            'query_id': idx,
            'query': query,
            'prediction': '',
            'reference': reference_answer,
            'precision@5': 0.0,
            'recall@5': 0.0,
            'mrr': 0.0,
            'ndcg@5': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bleu': 0.0,
            'semantic_similarity': 0.0,
            'llm_relevance': 0.0,
            'llm_accuracy': 0.0,
            'llm_completeness': 0.0,
            'llm_clarity': 0.0,
            'llm_overall': 0.0,
            'llm_explanation': ''
        }
        
        try:
            # Get system response
            agent_result = agent.invoke({"input": query})
            prediction = agent_result['output']
            result_dict['prediction'] = prediction
            
            # Extract retrieved article IDs from agent output
            retrieved_ids = []
            try:
                # Try to extract from intermediate_steps if available
                if 'intermediate_steps' in agent_result:
                    for step in agent_result.get('intermediate_steps', []):
                        if len(step) > 1:
                            retrieved_ids.extend(list(retrieval_evaluator.extract_article_ids(str(step[1]))))
                
                # Also try extracting from the output itself
                retrieved_ids.extend(list(retrieval_evaluator.extract_article_ids(str(agent_result))))
                
                # Remove duplicates while preserving order
                seen = set()
                retrieved_ids = [x for x in retrieved_ids if not (x in seen or seen.add(x))]
            except Exception:
                pass
            
            # Get relevant articles from test data
            relevant_articles = []
            try:
                if 'article_id' in row and pd.notna(row['article_id']):
                    relevant_articles = str(row['article_id']).split(',')
                elif 'article_ids' in row and pd.notna(row['article_ids']):
                    relevant_articles = row['article_ids'] if isinstance(row['article_ids'], list) else str(row['article_ids']).split(',')
                
                # Clean article IDs
                relevant_articles = [str(a).strip() for a in relevant_articles if str(a).strip()]
            except Exception:
                pass
            
            # Calculate retrieval metrics if we have both
            if retrieved_ids and relevant_articles:
                try:
                    result_dict['precision@5'] = retrieval_evaluator.precision_at_k(retrieved_ids, relevant_articles, k=5)
                    result_dict['recall@5'] = retrieval_evaluator.recall_at_k(retrieved_ids, relevant_articles, k=5)
                    result_dict['mrr'] = retrieval_evaluator.mean_reciprocal_rank(retrieved_ids, relevant_articles)
                    result_dict['ndcg@5'] = retrieval_evaluator.ndcg_at_k(retrieved_ids, relevant_articles, k=5)
                except Exception:
                    pass
            
            # Calculate response quality metrics with individual error handling
            try:
                rouge_scores = response_evaluator.calculate_rouge(prediction, reference_answer)
                result_dict['rouge1'] = rouge_scores.get('rouge1', 0.0)
                result_dict['rouge2'] = rouge_scores.get('rouge2', 0.0)
                result_dict['rougeL'] = rouge_scores.get('rougeL', 0.0)
            except Exception:
                pass
            
            try:
                result_dict['bleu'] = response_evaluator.calculate_bleu(prediction, reference_answer)
            except Exception:
                pass
            
            try:
                result_dict['semantic_similarity'] = response_evaluator.calculate_semantic_similarity(prediction, reference_answer)
            except Exception:
                pass
            
            # LLM as judge evaluation
            try:
                llm_eval = llm_judge.evaluate_response(query, prediction, reference_answer)
                result_dict['llm_relevance'] = llm_eval.get('relevance', 0.0)
                result_dict['llm_accuracy'] = llm_eval.get('accuracy', 0.0)
                result_dict['llm_completeness'] = llm_eval.get('completeness', 0.0)
                result_dict['llm_clarity'] = llm_eval.get('clarity', 0.0)
                result_dict['llm_overall'] = llm_eval.get('overall', 0.0)
                result_dict['llm_explanation'] = llm_eval.get('explanation', '')
            except Exception:
                pass
            
            print(f"  ✓ Completed (Semantic Sim: {result_dict['semantic_similarity']:.3f})")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            result_dict['prediction'] = f"ERROR: {str(e)}"
        
        results.append(result_dict)
        
        # Small delay to respect rate limits
        time.sleep(1)
    
    # Calculate aggregate metrics
    results_df = pd.DataFrame(results)
    
    # Ensure all required columns exist before accessing them
    required_cols = ['precision@5', 'recall@5', 'mrr', 'ndcg@5', 'rouge1', 'rouge2', 
                     'rougeL', 'bleu', 'semantic_similarity', 'llm_relevance', 
                     'llm_accuracy', 'llm_completeness', 'llm_clarity', 'llm_overall']
    
    for col in required_cols:
        if col not in results_df.columns:
            results_df[col] = 0.0
    
    if 'llm_explanation' not in results_df.columns:
        results_df['llm_explanation'] = ''
    
    aggregate_metrics = {
        'retrieval': {
            'avg_precision@5': results_df['precision@5'].mean(),
            'avg_recall@5': results_df['recall@5'].mean(),
            'avg_mrr': results_df['mrr'].mean(),
            'avg_ndcg@5': results_df['ndcg@5'].mean(),
        },
        'response_quality': {
            'avg_rouge1': results_df['rouge1'].mean(),
            'avg_rouge2': results_df['rouge2'].mean(),
            'avg_rougeL': results_df['rougeL'].mean(),
            'avg_bleu': results_df['bleu'].mean(),
            'avg_semantic_similarity': results_df['semantic_similarity'].mean(),
        },
        'llm_judge': {
            'avg_relevance': results_df['llm_relevance'].mean(),
            'avg_accuracy': results_df['llm_accuracy'].mean(),
            'avg_completeness': results_df['llm_completeness'].mean(),
            'avg_clarity': results_df['llm_clarity'].mean(),
            'avg_overall': results_df['llm_overall'].mean(),
        }
    }
    
    return {
        'detailed_results': results_df,
        'aggregate_metrics': aggregate_metrics,
        'total_queries': len(test_data),
        'successful_queries': len(results_df[results_df['prediction'] != ''])
    }
