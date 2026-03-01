import re
import os
import json
import torch
import random
import logging

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

# 设置logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


DEFINITIONS = {
    'Qwen3-Embedding-0.6B': {
        'cora': 'Represent the topic of this paper for clustering: ',
        'citeseer': 'Represent the topic of this paper for clustering: ',
        'pubmed': 'Represent the topic of this paper for clustering: ',
        'wikics': 'Represent the topic of this paper for clustering: ',
        'washington': 'Represent the role of this webpage for clustering: ',
        'wisconsin': 'Represent the role of this webpage for clustering: ',
        'texas': 'Represent the role of this webpage for clustering: ',
        'cornell': 'Represent the role of this webpage for clustering: ',
        'amazon': 'Represent the overall rating of this product for clustering: ',
        'actor': 'Represent the professional domain and cultural background of this actor for retrieval: ',
        'arxiv': 'Represent the topic of this paper for clustering: ',
    }
}

FILTER_PROMPT = {
    'cora': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of papers and determine whether they belong to the same research topic.
        
        Each paper belongs to one of the following predefined research topics: 
        [Rule Learning, Neural Networks, Case Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods]
        
        I will provide you with the title and abstract of each paper in the form of a JSON string:
            {"paper id_1": "title and abstract", "paper id_2": "title and abstract", ..., "paper id_n": "title and abstract"}
            
        Task:
        Analyze the provided papers and determine if they all belong to the same research topic.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the papers belong or do not belong to the same research topic", "answer": "true/false"}.
        2. Base your analysis solely on the provided titles and abstracts
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the research focus, methodology, and subject matter when making your determination
        5. If papers span multiple different research topics, answer should be "false"
        6. If all papers clearly belong to the same research topic, answer should be "true"
    """}],
    'citeseer': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of papers and determine whether they belong to the same research topic.
        
        Each paper belongs to one of the following predefined research topics: 
        [Agents, Machine Learning, Information Retrieval, Database, Human Computer Interaction, Artificial Intelligence.]
        
        I will provide you with the title and abstract of each paper in the form of a JSON string:
            {"paper id_1": "title and abstract", "paper id_2": "title and abstract", ..., "paper id_n": "title and abstract"}
            
        Task:
        Analyze the provided papers and determine if they all belong to the same research topic.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the papers belong or do not belong to the same research topic", "answer": "true/false"}.
        2. Base your analysis solely on the provided titles and abstracts
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the research focus, methodology, and subject matter when making your determination
        5. If papers span multiple different research topics, answer should be "false"
        6. If all papers clearly belong to the same research topic, answer should be "true"
    """}],
'arxiv': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of papers and determine whether they belong to the same research topic.
        
        I will provide you with the title and abstract of each paper in the form of a JSON string:
            {"paper id_1": "title and abstract", "paper id_2": "title and abstract", ..., "paper id_n": "title and abstract"}
            
        Task:
        Analyze the provided papers and determine if they all belong to the same research topic.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the papers belong or do not belong to the same research topic", "answer": "true/false"}.
        2. Base your analysis solely on the provided titles and abstracts
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the research focus, methodology, and subject matter when making your determination
        5. If papers span multiple different research topics, answer should be "false"
        6. If all papers clearly belong to the same research topic, answer should be "true"
    """}],
    'pubmed': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of papers and determine whether they belong to the same research topic.
        
        Each paper belongs to one of the following predefined research topics: 
        [Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2.]
        
        I will provide you with the title and abstract of each paper in the form of a JSON string:
            {"paper id_1": "title and abstract", "paper id_2": "title and abstract", ..., "paper id_n": "title and abstract"}
            
        Task:
        Analyze the provided papers and determine if they all belong to the same research topic.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the papers belong or do not belong to the same research topic", "answer": "true/false"}.
        2. Base your analysis solely on the provided titles and abstracts
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the research focus, methodology, and subject matter when making your determination
        5. If papers span multiple different research topics, answer should be "false"
        6. If all papers clearly belong to the same research topic, answer should be "true"
    """}],
    'wikics': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of wikipedia articles and determine whether they belong to the same technical field.
        
        Each article belongs to one of the following predefined technical fields: 
        [Computational linguistics, Databases, Operating systems, Computer architecture, Computer security, Internet protocols, Computer file systems, Distributed computing architecture, Web technology, Programming language topics.]
        
        I will provide you with the title and content of each article in the form of a JSON string:
            {"article id_1": "title and content", "article id_2": "title and content", ..., "article id_n": "title and content"}
            
        Task:
        Analyze the provided wikipedia articles and determine if they all belong to the same technical field.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the wikipedia articles belong or do not belong to the same technical field", "answer": "true/false"}
        2. Base your analysis solely on the provided titles and contents
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the primary technical field and core subject matter when making your determination
        5. If wikipedia articles span multiple different technical fields, answer should be "false"
        6. If all wikipedia articles clearly belong to the same technical field, answer should be "true"
    """}],
    'wisconsin': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of webpages from the University's computer science department and determine whether they all belong to the same webpage type.
        
        Each webpage belongs to one of the following predefined types:
        [Course, Project, Faculty, Student, Staff.]
        
        I will provide you with the title and content of each webpage in the form of a JSON string:
            {"webpage id_1": "title and content", "webpage id_2": "title and content", ..., "webpage id_n": "title and content"}
            
        Task:
        Analyze the provided webpages and determine if they all belong to the same type.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the webpages belong or do not belong to the same type", "answer": "true/false"}.
        2. Base your analysis solely on the provided titles and contents
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the functional role of the webpages when making your determination
        5. If webpages span multiple different types, answer should be "false"
        6. If all webpages clearly belong to the same type, answer should be "true"
    """}],
    'cornell': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of webpages from Cornell University and determine whether they all belong to the same webpage type.
        
        Each webpage belongs to one of the following predefined types:
        [Course, Project, Faculty, Student, Staff.]
        
        I will provide you with the title and content of each webpage in the form of a JSON string:
            {"webpage id_1": "title and content", "webpage id_2": "title and content", ..., "webpage id_n": "title and content"}
            
        Task:
        Analyze the provided webpages and determine if they all belong to the same type.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the webpages belong or do not belong to the same type", "answer": "true/false"}.
        2. Base your analysis solely on the provided titles and contents
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the functional role of the webpages when making your determination
        5. If webpages span multiple different types, answer should be "false"
        6. If all webpages clearly belong to the same type, answer should be "true"
    """}],
    'texas': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of webpages from the University of Texas at Austin website and determine whether they all belong to the same webpage type.
        
        Each webpage belongs to one of the following predefined types:
        [Course, Project, Faculty, Student, Staff.]
        
        I will provide you with the title and content of each webpage in the form of a JSON string:
            {"webpage id_1": "title and content", "webpage id_2": "title and content", ..., "webpage id_n": "title and content"}
            
        Task:
        Analyze the provided webpages and determine if they all belong to the same type.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the webpages belong or do not belong to the same type", "answer": "true/false"}.
        2. Base your analysis solely on the provided titles and contents
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the functional role of the webpages when making your determination
        5. If webpages span multiple different types, answer should be "false"
        6. If all webpages clearly belong to the same type, answer should be "true"
    """}],
    'actor': [{"role": "system", "content": """
        You will act as an assistant to help me analyze a batch of actors and determine whether they all belong to the same category.
        
        Each actor belongs to one of the following predefined categories:
        [American film actors (only), American film actors and American television actors, American television actors and American stage actors, English actors, Canadian actors.]
        
        I will provide you with the name and keywords of each actor in the form of a JSON string:
            {"actor id_1": "name and keywords", "actor id_2": "name and keywords", ..., "actor id_n": "name and keywords"}
            
        Task:
        Analyze the provided actors and determine if they all belong to the same category.
                
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {"explanation": "detailed reasoning explaining why the actors belong or do not belong to the same category", "answer": "true/false"}.
        2. Base your analysis solely on the provided names and keywords
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
        4. Consider the professional domain and cultural background of the actors when making your determination
        5. If actors span multiple different categories, answer should be "false"
        6. If all actors clearly belong to the same category, answer should be "true"
    """}],
}

EXPLAINATION_COT_PROMPT = {
    'cora': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for citations between two academic papers.
        
        I will provide you with the title and abstract of each paper in the form of a JSON string:
            {"title and abstract of paper 1", "title and abstract of paper 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a citation relationship between these two papers."}
        2. Base your analysis solely on the provided titles and abstracts
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
    'citeseer': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for citations between two academic papers.
        
        I will provide you with the title and abstract of each paper in the form of a JSON string:
            {"title and abstract of paper 1", "title and abstract of paper 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a citation relationship between these two papers."}
        2. Base your analysis solely on the provided titles and abstracts
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
    'pubmed': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for citations between two academic papers.
        
        I will provide you with the title and abstract of each paper in the form of a JSON string:
            {"title and abstract of paper 1", "title and abstract of paper 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a citation relationship between these two papers."}
        2. Base your analysis solely on the provided titles and abstracts
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
    'wikics': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for hyperlink between two wikipedia articles.
        
        I will provide you with the title and content of each wikipedia article in the form of a JSON string:
            {"title and content of article 1", "title and content of article 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a hyperlink relationship between these two wikipedia articles."}
        2. Base your analysis solely on the provided titles and contents
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
    'arxiv': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for citations between two academic papers.
        
        I will provide you with the title and abstract of each paper in the form of a JSON string:
            {"title and abstract of paper 1", "title and abstract of paper 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a citation relationship between these two papers."}
        2. Base your analysis solely on the provided titles and abstracts
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
    'wisconsin': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for hyperlink between two webpages from the University's computer science department website.
        
        I will provide you with the title and content of each webpage in the form of a JSON string:
            {"title and content of webpage 1", "title and content of webpage 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a hyperlink relationship between these two webpages."}
        2. Base your analysis solely on the provided titles and contents
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
    'texas': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for hyperlink between two webpages from the University of Texas at Austin website.
        
        I will provide you with the title and content of each webpage in the form of a JSON string:
            {"title and content of webpage 1", "title and content of webpage 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a hyperlink relationship between these two webpages."}
        2. Base your analysis solely on the provided titles and contents
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
    'cornell': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for hyperlink between two webpages from the Cornell University website.
        
        I will provide you with the title and content of each webpage in the form of a JSON string:
            {"title and content of webpage 1", "title and content of webpage 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a hyperlink relationship between these two webpages."}
        2. Base your analysis solely on the provided titles and contents
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
    'actor': [{"role": "system", "content": """
        You will act as an assistant to help me infer the potential reason for the two actors being linked in the actor knowledge graph. 
        
        I will provide you with the name and keywords of each actor in the form of a JSON string:
            {"name and keywords of actor 1", "name and keywords of actor 2"}
        
        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
        {"reasoning": "Use 50 words to briefly reason why there is a link relationship between these two actors."}
        2. Base your analysis solely on the provided names and keywords
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided
    """}],
}


SEGMENTATION_PROMPT = {
    'cora': [{"role": "system", "content": """
        You will act as an assistant to help me segment the citation path of academic papers into continuous sub paths, so that each sub path contains semantically coherent papers (i.e. belonging to the same research topic).

        Each paper belongs to one of the following predefined research topics: 
        [Rule Learning, Neural Networks, Case Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods]

        I will provide you with the title and abstract of each paper in the path in the form of a JSON string:
            {"paper 1": "title and abstract", "paper 2": "title and abstract", ..., "paper n": "title and abstract"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"paper_ids": ["paper 1", "paper 2", ...]},
                    {"paper_ids": ["paper 3", "paper 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided titles and abstracts.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all papers are about the similar research topic.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same topic (e.g., different algorithms for Bayesian networks) should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two papers belong to the similar topic, they must appear in the same segment.
    """}],
    'citeseer': [{"role": "system", "content": """
        You will act as an assistant to help me segment the citation path of academic papers into continuous sub paths, so that each sub path contains semantically coherent papers (i.e. belonging to the same research topic).

        Each paper belongs to one of the following predefined research topics: 
        [Agents, Machine Learning, Information Retrieval, Database, Human Computer Interaction, Artificial Intelligence.]

        I will provide you with the title and abstract of each paper in the path in the form of a JSON string:
            {"paper 1": "title and abstract", "paper 2": "title and abstract", ..., "paper n": "title and abstract"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"paper_ids": ["paper 1", "paper 2", ...]},
                    {"paper_ids": ["paper 3", "paper 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided titles and abstracts.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all papers are about the similar research topic.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same topic (e.g., different algorithms for Bayesian networks) should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two papers belong to the similar topic, they must appear in the same segment.
    """}],
    'arxiv': [{"role": "system", "content": """
        You will act as an assistant to help me segment the citation path of academic papers into continuous sub paths, so that each sub path contains semantically coherent papers (i.e. belonging to the same research topic).

        I will provide you with the title and abstract of each paper in the path in the form of a JSON string:
            {"paper 1": "title and abstract", "paper 2": "title and abstract", ..., "paper n": "title and abstract"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"paper_ids": ["paper 1", "paper 2", ...]},
                    {"paper_ids": ["paper 3", "paper 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided titles and abstracts.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all papers are about the similar research topic.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same topic should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two papers belong to the similar topic, they must appear in the same segment.
    """}],
    'pubmed': [{"role": "system", "content": """
        You will act as an assistant to help me segment the citation path of academic papers into continuous sub paths, so that each sub path contains semantically coherent papers (i.e. belonging to the same research topic).

        Each paper belongs to one of the following predefined research topics: 
        [Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2.]

        I will provide you with the title and abstract of each paper in the path in the form of a JSON string:
            {"paper 1": "title and abstract", "paper 2": "title and abstract", ..., "paper n": "title and abstract"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"paper_ids": ["paper 1", "paper 2", ...]},
                    {"paper_ids": ["paper 3", "paper 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided titles and abstracts.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all papers are about the similar research topic.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same topic (e.g., one paper on insulin resistance and another on beta-cell dysfunction in Type 2 diabetes) should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two papers belong to the similar topic, they must appear in the same segment.
    """}],
    'wikics': [{"role": "system", "content": """
        You will act as an assistant to help me segment the hyperlink path of wikipedia articles into continuous sub paths, so that each sub path contains semantically coherent wikipedia articles (i.e. belonging to the same technical field).

        Each wikipedia article belongs to one of the following predefined technical fields: 
        [Computational linguistics, Databases, Operating systems, Computer architecture, Computer security, Internet protocols, Computer file systems, Distributed computing architecture, Web technology, Programming language topics.]

        I will provide you with the title and content of each wikipedia article in the path in the form of a JSON string:
            {"article 1": "title and content", "article 2": "title and content", ..., "article n": "title and content"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"article_ids": ["article 1", "article 2", ...]},
                    {"article_ids": ["article 3", "article 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided titles and contents.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all wikipedia articles are about the similar technical field.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same technical field (e.g., one wikipedia article on file allocation methods and another on journaling in computer file systems) should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two wikipedia articles belong to the similar technical field, they must appear in the same segment.
    """}],
    'wisconsin': [{"role": "system", "content": """
        You will act as an assistant to help me segment a hyperlink path of webpages from the University's computer science department website into continuous subpaths, such that each subpath contains semantically coherent webpages belonging to the same type.

        Each webpage belongs to one of the following predefined types: 
        [Course, Project, Faculty, Student, Staff.]

        I will provide you with the title and content of each webpage in the path in the form of a JSON string:
            {"webpage 1": "title and content", "webpage 2": "title and content", ..., "webpage n": "title and content"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"webpage_ids": ["webpage 1", "webpage 2", ...]},
                    {"webpage_ids": ["webpage 3", "webpage 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided titles and contents.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all webpages belong to the same type.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same type (e.g., a webpage for an undergraduate course and another for a graduate course, both clearly labeled as courses) should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two webpages belong to the same type, they must appear in the same segment.
    """}],
    # 'cornell': [{"role": "system", "content": """
    #     You will act as an assistant to help me segment a hyperlink path of webpages from the Cornell University website into continuous subpaths, such that each subpath contains semantically coherent webpages belonging to the same type.

    #     Each webpage belongs to one of the following predefined types: 
    #     [Course, Project, Faculty, Student, Staff.]

    #     I will provide you with the title and content of each webpage in the path in the form of a JSON string:
    #         {"webpage 1": "title and content", "webpage 2": "title and content", ..., "webpage n": "title and content"}

    #     Requirements:
    #     1. Please provide your response in the strict JSON format, following this structure:
    #         {   
    #             "explanation": "Use 200 words to explain in detail where and why the path should be split.",
    #             "segments": [
    #                 {"webpage_ids": ["webpage 1", "webpage 2", ...]},
    #                 {"webpage_ids": ["webpage 3", "webpage 4", ...]},
    #                 ...
    #             ]
    #         }
    #     2. Base your analysis solely on the provided titles and contents.
    #     3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
    #     4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all webpages belong to the same type.
    #     5. Do not output any text before or after the JSON object.
    #     6. Minor variations within the same type (e.g., a webpage for an undergraduate course and another for a graduate course, both clearly labeled as courses) should not trigger a split.
    #     7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two webpages belong to the same type, they must appear in the same segment.
    # """}],
     'cornell': [{"role": "system", "content": """
        You will act as an assistant to help me segment a hyperlink path of webpages from the Cornell University website into continuous subpaths, such that each subpath contains semantically coherent webpages belonging to the same type.

        I will provide you with the title and content of each webpage in the path in the form of a JSON string:
            {"webpage 1": "title and content", "webpage 2": "title and content", ..., "webpage n": "title and content"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"webpage_ids": ["webpage 1", "webpage 2", ...]},
                    {"webpage_ids": ["webpage 3", "webpage 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided titles and contents.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all webpages belong to the same type.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same type (e.g., a webpage for an undergraduate course and another for a graduate course, both clearly labeled as courses) should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two webpages belong to the same type, they must appear in the same segment.
    """}],
    'texas': [{"role": "system", "content": """
        You will act as an assistant to help me segment a hyperlink path of webpages from the University of Texas at Austin website into continuous subpaths, such that each subpath contains semantically coherent webpages belonging to the same type.

        Each webpage belongs to one of the following predefined types: 
        [Course, Project, Faculty, Student, Staff.]

        I will provide you with the title and content of each webpage in the path in the form of a JSON string:
            {"webpage 1": "title and content", "webpage 2": "title and content", ..., "webpage n": "title and content"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"webpage_ids": ["webpage 1", "webpage 2", ...]},
                    {"webpage_ids": ["webpage 3", "webpage 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided titles and contents.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all webpages belong to the same type.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same type (e.g., a webpage for an undergraduate course and another for a graduate course, both clearly labeled as courses) should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two webpages belong to the same type, they must appear in the same segment.
    """}],
    'actor': [{"role": "system", "content": """
        You will act as an assistant to help me segment the social relationship path of actors into continuous sub paths, so that each sub path contains actors from the same country.

        Each actor belongs to one of the following predefined categories: 
        [American film actors (only), American film actors and American television actors, American television actors and American stage actors, English actors, Canadian actors.]

        I will provide you with the name and keywords of each actor in the path in the form of a JSON string:
            {"actor 1": "name and keywords", "actor 2": "name and keywords", ..., "actor n": "name and keywords"}

        Requirements:
        1. Please provide your response in the strict JSON format, following this structure:
            {   
                "explanation": "Use 200 words to explain in detail where and why the path should be split.",
                "segments": [
                    {"actor_ids": ["actor 1", "actor 2", ...]},
                    {"actor_ids": ["actor 3", "actor 4", ...]},
                    ...
                ]
            }
        2. Base your analysis solely on the provided names and keywords.
        3. Do not introduce external knowledge or make subjective assumptions beyond the content provided.
        4. Split the path into **contiguous segments** (preserving the original order) such that within each segment, all actors are belong to the same category.
        5. Do not output any text before or after the JSON object.
        6. Minor variations within the same category (e.g., American film actors comes from different states) should not trigger a split.
        7. The `segments` field MUST exactly match the segmentation described in the `explanation`. If you state that two actors belong to the same category, they must appear in the same segment.
    """}],
}

def split_high_diversity_paths(llm_model_path, high_diversity_paths, raw_texts, dataset_name):
    model = llm_model_path
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, padding_side='left')
    sampling_params = SamplingParams(temperature=0, top_p=0.8, repetition_penalty=1.05, max_tokens=25000)
    llm = LLM(model=model, trust_remote_code=True, dtype=torch.bfloat16, max_model_len=42000)

    # 根据数据集选择节点名称
    if dataset_name in ['cora', 'citeseer', 'pubmed', 'arxiv']:
        node_name = 'paper'
    elif dataset_name == 'wikics':
        node_name = 'article'
    elif dataset_name == 'actor':
        node_name = 'actor'
    elif dataset_name in ['wisconsin', 'washington', 'texas', 'cornell']:
        node_name = 'webpage'
    else:
        node_name = 'paper'  # 默认使用paper

    inputs = []
    processed_paths = []

    for i, path_item in enumerate(tqdm(high_diversity_paths, desc="Preparing segmentation prompts")):
        path = path_item['path']
        nodes_dict = {}
        for j, node_id in enumerate(path):
            if 0 <= node_id < len(raw_texts):
                nodes_dict[f"{node_name} {j+1}"] = raw_texts[node_id]
            else:
                nodes_dict[f"{node_name} {j+1}"] = f"Text for node {node_id} not available."
        user_content = json.dumps(nodes_dict, ensure_ascii=False, indent=2)
        prompt = SEGMENTATION_PROMPT[dataset_name] + [{"role": "user", "content": user_content}]
        inputs.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
        if i < 5:
            logger.info(f"Input #{i+1} data (complete prompt):\nSystem: {prompt[0]}\nUser: {prompt[1]}")
        processed_paths.append(path_item)

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    results = []
    for i, output in enumerate(outputs):
        path_item = processed_paths[i]
        generated = output.outputs[0].text.strip()

        segmentation_result = {
            "segments": None,
            "explanation": "Failed to parse LLM response",
            "raw_output": generated
        }

        try:
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                segmentation_result["explanation"] = parsed.get("explanation", "")
                
                segments_raw = parsed.get("segments", [])
                if isinstance(segments_raw, list) and segments_raw:
                    new_segments = []
                    for seg in segments_raw:
                        node_ids = []
                        
                        # 处理两种情况：
                        # 1. seg 是字典，包含 paper_ids 字段
                        # 2. seg 是列表，直接是节点ID列表
                        if isinstance(seg, dict):
                            paper_ids = seg.get("paper_ids", [])
                            for pid in paper_ids:
                                try:
                                    idx = int(pid.split()[1]) - 1
                                    if 0 <= idx < len(path_item['path']):
                                        node_ids.append(path_item['path'][idx])
                                except (ValueError, IndexError, AttributeError):
                                    continue
                        elif isinstance(seg, list):
                            # 如果 seg 直接是节点ID列表，检查是否在原始路径中
                            for node_id in seg:
                                if node_id in path_item['path']:
                                    node_ids.append(node_id)
                        
                        if node_ids:
                            new_segments.append(node_ids)
                    
                    if new_segments:
                        segmentation_result["segments"] = new_segments
                    else:
                        segmentation_result["segments"] = [[n] for n in path_item['path']]
                else:
                    segmentation_result["segments"] = [path_item['path']]

        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            segmentation_result["explanation"] = f"Parsing error: {str(e)}"

        path_item["segmentation_result"] = segmentation_result
        results.append(path_item)

    return results


def generate_edge_explanations(edge_index, raw_texts, llm_model_path, dataset_name):
    model = llm_model_path
    edge_list = edge_index.t().tolist()

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, padding_side='left')
    sampling_params = SamplingParams(temperature=0, top_p=0.8, repetition_penalty=1.05, max_tokens=25000)
    llm = LLM(model=model, trust_remote_code=True, dtype=torch.bfloat16, max_model_len=50000)

    inputs = []

    for i, (src, dst) in enumerate(tqdm(edge_list, desc="Preparing prompts")):
        text1 = raw_texts[src] if 0 <= src < len(raw_texts) else f"Text for node {src} not available."
        text2 = raw_texts[dst] if 0 <= dst < len(raw_texts) else f"Text for node {dst} not available."
        user_content = json.dumps([text1, text2], ensure_ascii=False, indent=2)
        prompt = EXPLAINATION_COT_PROMPT[dataset_name] + [{"role": "user", "content": user_content}]
        inputs.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
        if i == 0:
            logger.info(f"First input data (complete prompt):\nSystem: {prompt[0]}\nUser: {prompt[1]}")

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    results = []
    for (src, dst), output in zip(edge_list, outputs):
        generated = output.outputs[0].text.strip()

        reasoning = "Failed to parse LLM response"
        parse_success = False
        
        try:
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if "reasoning" in parsed and isinstance(parsed["reasoning"], str):
                    reasoning = parsed["reasoning"].strip()
                    parse_success = True
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        result = {
            "edge": [src, dst],
            "reasoning": reasoning
        }
        
        if not parse_success:
            result["raw_response"] = generated

        results.append(result)

    return results


def low_diversity_filter(llm_model_path, low_diversity_paths, raw_texts, args):
    model = llm_model_path
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, padding_side='left')
    sampling_params = SamplingParams(temperature=0, top_p=0.8, repetition_penalty=1.05, max_tokens=20000)
    llm = LLM(model=model, trust_remote_code=True, dtype=torch.bfloat16, max_model_len=50000)

    # 根据数据集选择节点名称
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'arxiv']:
        node_name = 'paper'
    elif args.dataset == 'wikics':
        node_name = 'article'
    elif args.dataset == 'actor':
        node_name = 'actor'
    elif args.dataset in ['wisconsin', 'washington', 'texas', 'cornell']:
        node_name = 'webpage'
    else:
        node_name = 'paper'  # 默认使用paper

    inputs = []
    processed_paths = []
    
    for i, path_item in enumerate(tqdm(low_diversity_paths, desc="准备筛选输入")):
        path = path_item['path']

        nodes_dict = {}
        for j, node_id in enumerate(path):
            node_text = raw_texts[node_id] if node_id < len(raw_texts) else f"Node {node_id} text not found"
            nodes_dict[f"{node_name} id_{j+1}"] = node_text

        user_content = json.dumps(nodes_dict, ensure_ascii=False, indent=2)
        input_data = FILTER_PROMPT[args.dataset] + [{"role": "user", "content": user_content}]
        inputs.append(tokenizer.apply_chat_template(input_data, tokenize=False, add_generation_prompt=True))
        if i == 0:
            logger.info(f"First input data (complete prompt):\nSystem: {input_data[0]}\nUser: {input_data[1]}")
        processed_paths.append(path_item)

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    filtered_results = []
    for i, output in enumerate(tqdm(outputs, desc="处理筛选结果")):
        generated_text = output.outputs[0].text
        path_item = processed_paths[i]
        
        try:
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                response_data = json.loads(json_str)
                path_item['llm_filter_result'] = response_data
                filtered_results.append(path_item)
            else:
                path_item['llm_filter_result'] = {
                    'explanation': 'Failed to parse LLM response',
                    'answer': 'false',
                    'raw_output': generated_text
                }
                filtered_results.append(path_item)
                
        except json.JSONDecodeError:
            path_item['llm_filter_result'] = {
                'explanation': 'Failed to parse LLM response',
                'answer': 'false',
                'raw_output': generated_text
            }
            filtered_results.append(path_item)

    return filtered_results


def split_low_diversity_paths(low_diversity_paths):

    def split_path_at_position(path):
        if len(path) <= 1:
            return path, []
        
        num_edges = len(path) - 1
        if num_edges == 0:
            return path, []
        
        if num_edges == 1:
            # 只有一条边：分割成两个孤立节点
            return [path[0]], [path[1]]
        else:
            # 按边数对半分
            split_edge_pos = num_edges // 2
            path_first = path[:split_edge_pos + 1]
            path_second = path[split_edge_pos + 1:]
            return path_first, path_second
    
    results = []
    for path_item in tqdm(low_diversity_paths, desc="Splitting low diversity paths"):
        path = path_item['path']
        path_first, path_second = split_path_at_position(path)
        
        segments = [path_first, path_second] if path_second else [path_first]
        
        segmentation_result = {
            "segments": segments
        }

        result = {
            "path": path,
            "labels": path_item.get("labels", []),
            "diversity_score": path_item.get("diversity_score", 0.0),
            "segmentation_result": segmentation_result
        }
        
        results.append(result)
    
    return results


def set_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This might slow down your training.


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, prompt_text=None, max_length=8192):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        if self.prompt_text:
            input_text = self.prompt_text + sentence
        else:
            input_text = sentence
            
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def generate_features_with_llm(args, sentences):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_name,
        padding_side='left'
    )
    model = AutoModel.from_pretrained(
        args.embedding_model_name,
    ).to(device)
    print(f"Encoding {len(sentences)} sentences with batched inference...")

    prompt_text = None
    if args.prompt:
        print(f'Using prompt for dataset: {args.dataset}')
        prompt_text = DEFINITIONS[args.prompt][args.dataset]

    dataset = SentenceDataset(
        sentences,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_length=2048
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    batch_size = 32
    actual_batch_size = batch_size * torch.cuda.device_count()
    data_loader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        collate_fn=data_collator,
        pin_memory=True
    )

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Encoding", mininterval=10, disable=len(sentences) < 128):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.amp.autocast('cuda'):
                outputs = model(**batch)
                # embeds = last_token_pool(outputs.last_hidden_state, batch['attention_mask'])[:, : args.emdedding_dim]
                embeds = last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
                embeds = F.normalize(embeds, p=2, dim=1)
                all_embeddings.append(embeds)

    corpus_embeddings = torch.cat(all_embeddings, dim=0)

    return corpus_embeddings


def diversity_score(path_nodes, features):
    if len(path_nodes) < 2:
        return 0.0

    path_indices = torch.tensor(path_nodes, dtype=torch.long, device=features.device)
    node_embeds = features[path_indices]
    node_embeds = F.normalize(node_embeds, p=2, dim=1)
    
    centroid = node_embeds.mean(dim=0, keepdim=True)
    var = ((node_embeds - centroid) ** 2).sum(dim=1).mean()
    
    L = len(path_nodes)

    normalized_var = var / (L ** 0.5)
    
    return float(normalized_var.item())





def plot_diversity_label_consistency(
    scores,
    dataset_name="unknown",
    save_path=None,
    num_bins=40,
    normalize_scores=True,  # 新增参数：是否归一化多样性分数到 [0,1]
):
    if not scores:
        print("No scores to plot.")
        return

    diversity_scores = np.array([item['diversity_score'] for item in scores])
    
    if normalize_scores:
        min_score = diversity_scores.min()
        max_score = diversity_scores.max()
        if max_score - min_score > 1e-8:
            diversity_scores = (diversity_scores - min_score) / (max_score - min_score)
        else:
            diversity_scores = np.zeros_like(diversity_scores)
        score_range = (0.0, 1.0)
    else:
        score_range = (diversity_scores.min(), diversity_scores.max())

    is_homogeneous = [len(set(item['labels'])) == 1 for item in scores]

    bins = np.linspace(score_range[0], score_range[1], num_bins + 1)
    bin_indices = np.digitize(diversity_scores, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    total_counts = np.zeros(num_bins, dtype=int)
    homogeneous_counts = np.zeros(num_bins, dtype=int)

    for i, bin_idx in enumerate(bin_indices):
        total_counts[bin_idx] += 1
        if is_homogeneous[i]:
            homogeneous_counts[bin_idx] += 1

    heterogeneous_counts = total_counts - homogeneous_counts

    with np.errstate(divide='ignore', invalid='ignore'):
        homo_ratio = np.divide(homogeneous_counts, total_counts, dtype=float)
        hetero_ratio = np.divide(heterogeneous_counts, total_counts, dtype=float)

    x = np.arange(num_bins)
    width = 0.6

    fig, ax = plt.subplots(figsize=(20, 6))
    bars1 = ax.bar(x, homogeneous_counts, width, label='All Labels Same', color='#4CAF50')
    bars2 = ax.bar(x, heterogeneous_counts, width, bottom=homogeneous_counts, label='Mixed Labels', color='#F44336')

    max_count = max(total_counts) if total_counts.any() else 1

    for i in range(num_bins):
        if total_counts[i] == 0:
            continue

        # 绿色部分顶部位置
        green_top = homogeneous_counts[i]
        # 红色部分顶部位置
        red_top = total_counts[i]

        # 绿色部分百分比（写在绿色柱子顶部）
        if homogeneous_counts[i] > 0:
            ax.text(i, red_top + max_count * 0.005, f'{homo_ratio[i]*100:.1f}%', 
                    ha='center', va='bottom', color='#4CAF50', fontweight='bold', fontsize=4)

        # 红色部分百分比（写在红色柱子顶部）
        if heterogeneous_counts[i] > 0:
            ax.text(i, red_top + 4 * max_count * 0.005, f'{hetero_ratio[i]*100:.1f}%', 
                    ha='center', va='bottom', color='#F44336', fontweight='bold', fontsize=4)

        # 总数 n=xxx（可选：放在最顶部，字体小一点）
        ax.text(i, red_top + 10 * max_count * 0.005, f'n={total_counts[i]}', 
                ha='center', va='bottom', fontsize=5, color='black')

    # 坐标轴设置（保持不变）
    ax.set_xlabel('Diversity Score (Normalized to [0,1])' if normalize_scores else 'Diversity Score')
    ax.set_ylabel('Number of Paths')
    title = f'Diversity vs Label Consistency (Dataset: {dataset_name})'
    if normalize_scores:
        title += ' [Scores Normalized to 0-1]'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'[{bins[i]:.2f}, {bins[i+1]:.2f})' for i in range(num_bins)], rotation=45)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def build_natural_language_paths(high_diversity_split_results, low_diversity_split_results, filtered_low_diversity_paths, edge_explanations, raw_texts, high_output_path, low_output_path, dataset_name):

    edge_dict = {}
    for edge_item in edge_explanations:
        src, dst = edge_item['edge']
        reasoning = edge_item['reasoning']
        
        # 跳过解析失败的边解释
        if reasoning == "Failed to parse LLM response":
            continue
        
        key = f"{src}_{dst}"
        edge_dict[key] = reasoning

        reverse_key = f"{dst}_{src}"
        edge_dict[reverse_key] = reasoning

    def get_node_text(node_id):
        if 0 <= node_id < len(raw_texts):
            return raw_texts[node_id]
        return ""
    
    def get_edge_explanation(node1_id, node2_id):
        key1 = f"{node1_id}_{node2_id}"
        key2 = f"{node2_id}_{node1_id}"
        
        if key1 in edge_dict:
            return edge_dict[key1]
        elif key2 in edge_dict:
            return edge_dict[key2]
        else:
            return ""
    
    def build_segment_nl(segment):
        if len(segment) == 0:
            return ""
        elif len(segment) == 1:
            return get_node_text(segment[0])
        else:
            parts = []
            
            parts.append(get_node_text(segment[0]))
            
            for i in range(len(segment) - 1):
                current_node = segment[i]
                next_node = segment[i + 1]
                
                edge_exp = get_edge_explanation(current_node, next_node)
                if edge_exp:
                    parts.append(edge_exp)
                
                parts.append(get_node_text(next_node))
            
            return "|".join(parts)
    
    high_nl_paths = []
    max_high_chars = 0
    
    for item in tqdm(high_diversity_split_results, desc="构建高多样性自然语言路径"):
        path = item['path']
        segments = item.get('segmentation_result', {}).get('segments', [])

        if not segments:
            continue
        
        nl_segments = []
        for segment in segments:
            nl_text = build_segment_nl(segment)
            if nl_text:
                nl_segments.append(nl_text)
                if len(nl_text) > max_high_chars:
                    max_high_chars = len(nl_text)
        
        high_nl_paths.append({
            "original_path": path,
            "segments": segments,
            "natural_language_segments": nl_segments,
            "path_length": len(path),
            "segment_count": len(segments),
            "diversity_score": item.get('diversity_score', 0.0)
        })

    low_nl_paths = []
    max_low_chars = 0
    
    for item in tqdm(low_diversity_split_results, desc="构建低多样性自然语言路径"):
        path = item['path']
        segments = item.get('segmentation_result', {}).get('segments', [])
        
        if not segments:
            continue
        
        nl_segments = []
        for segment in segments:
            nl_text = build_segment_nl(segment)
            if nl_text:
                nl_segments.append(nl_text)
                if len(nl_text) > max_low_chars:
                    max_low_chars = len(nl_text)
        
        low_nl_paths.append({
            "original_path": path,
            "segments": segments,
            "natural_language_segments": nl_segments,
            "path_length": len(path),
            "segment_count": len(segments),
            "diversity_score": item.get('diversity_score', 0.0)
        })

    high_result = {
        "dataset": dataset_name,
        "diversity_type": "high",
        "total_paths": len(high_nl_paths),
        "max_segment_chars": max_high_chars,
        "results": high_nl_paths
    }
    
    os.makedirs(os.path.dirname(high_output_path), exist_ok=True)
    with open(high_output_path, 'w', encoding='utf-8') as f:
        json.dump(high_result, f, ensure_ascii=False, indent=2)
    
    low_result = {
        "dataset": dataset_name,
        "diversity_type": "low",
        "total_paths": len(low_nl_paths),
        "max_segment_chars": max_low_chars,
        "results": low_nl_paths
    }
    
    os.makedirs(os.path.dirname(low_output_path), exist_ok=True)
    with open(low_output_path, 'w', encoding='utf-8') as f:
        json.dump(low_result, f, ensure_ascii=False, indent=2)
    
    return {"high_diversity": high_result,"low_diversity": low_result}