"""
Configuration file for embedding models.
Add or modify models here to compare different medical embedding models.
"""

# General-purpose model (baseline for comparison)
GENERAL_MODEL = {
    'name': 'bert-base-uncased',
    'description': 'BERT base model - general-purpose transformer',
    'dimensions': 768
}

# Medical/Clinical embedding models
MEDICAL_MODELS = {
    'pubmedbert': {
        'name': 'pritamdeka/S-PubMedBert-MS-MARCO',
        'description': 'PubMedBERT fine-tuned on MS-MARCO (medical literature)',
        'dimensions': 768
    },
    'biobert': {
        'name': 'dmis-lab/biobert-base-cased-v1.2',
        'description': 'BioBERT - pre-trained on biomedical literature',
        'dimensions': 768
    },
    'clinicalbert': {
        'name': 'emilyalsentzer/Bio_ClinicalBERT',
        'description': 'ClinicalBERT - trained on clinical notes',
        'dimensions': 768
    },
    'sapbert': {
        'name': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        'description': 'SapBERT - self-alignment pre-training for medical entities',
        'dimensions': 768
    },
    'bluebert': {
        'name': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        'description': 'BlueBERT - trained on PubMed and MIMIC-III',
        'dimensions': 768
    }
}

# Default medical model to use
DEFAULT_MEDICAL_MODEL = 'pubmedbert'

# Vector database settings
VECTOR_DB_CONFIG = {
    'base_dir': 'vector_dbs',
    'batch_size': 100,
    'max_terms': 5000  # Limit for demo purposes, set to None for full dataset
}
