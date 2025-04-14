import json
import pandas as pd
import numpy as np
import networkx as nx
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict
import torch
import os

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Constants
MAX_SEQ_LEN = 128  # Maximum sequence length for BERT
DATA_DIR = './data'  # Directory containing train.jsonl, validation.jsonl, test.jsonl
OUTPUT_DIR = './preprocessed_data'  # Directory to save preprocessed data
FILES = ['train.jsonl', 'validation.jsonl', 'test.jsonl']

def filter_empty_records(records):
    """Filter out records with empty messages."""
    filtered_records = [
        record for record in records if len(record['messages']) > 0
    ]
    print(f"Filtered {len(records) - len(filtered_records)} empty records. {len(filtered_records)} records remain.")
    return filtered_records

def load_jsonl(file_path):
    """Load JSONL file and return a list of records."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def tokenize_messages(messages, max_len=MAX_SEQ_LEN):
    """Tokenize messages using BERT tokenizer."""
    encodings = tokenizer(
        messages,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encodings['input_ids'], encodings['attention_mask']

def normalize_metadata(records):
    """Normalize metadata features (categorical and numerical)."""
    metadata = {
        'game_score': [],
        'game_score_delta': [],
        'seasons': [],
        'years': [],
        'players': [],
        'game_id': [],
        'absolute_message_index': [],
        'relative_message_index': []
    }
    
    for record in records:
        for i in range(len(record['messages'])):
            metadata['game_score'].append(record['game_score'][i])
            metadata['game_score_delta'].append(record['game_score_delta'][i])
            metadata['seasons'].append(record['seasons'][i])
            metadata['years'].append(record['years'][i])
            # Join players list into a single string for encoding
            metadata['players'].append('_'.join(sorted(record['players'])))
            metadata['game_id'].append(record['game_id'])
            metadata['absolute_message_index'].append(record['absolute_message_index'][i])
            metadata['relative_message_index'].append(record['relative_message_index'][i])
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata)
    
    # Encode categorical variables
    season_encoder = LabelEncoder()
    year_encoder = LabelEncoder()
    players_encoder = LabelEncoder()
    df['seasons'] = season_encoder.fit_transform(df['seasons'])
    df['years'] = year_encoder.fit_transform(df['years'])
    df['players'] = players_encoder.fit_transform(df['players'])
    
    # Encode game_id (if needed for model, otherwise drop it later)
    game_id_encoder = LabelEncoder()
    df['game_id'] = game_id_encoder.fit_transform(df['game_id'])
    
    # Scale numerical features
    scaler = StandardScaler()
    df[['game_score', 'game_score_delta', 'absolute_message_index', 'relative_message_index', 'seasons', 'years', 'players', 'game_id']] = scaler.fit_transform(
        df[['game_score', 'game_score_delta', 'absolute_message_index', 'relative_message_index', 'seasons', 'years', 'players', 'game_id']]
    )
    
    return df, season_encoder, year_encoder, players_encoder, game_id_encoder, scaler

def build_interaction_graph(records):
    """Build a player interaction graph per game phase, weighted by truthfulness."""
    graphs = defaultdict(nx.DiGraph)  # {game_id_phase: graph}
    
    for record in records:
        game_id = record['game_id']
        for i, (speaker, receiver, sender_label, season, year) in enumerate(zip(
            record['speakers'], record['receivers'], record['sender_labels'], record['seasons'], record['years']
        )):
            # Create a unique phase identifier (game_id + year + season)
            phase = f"{game_id}_{year}_{season}"
            
            # Initialize graph if not exists
            if phase not in graphs:
                graphs[phase] = nx.DiGraph()
            
            # Add nodes
            graphs[phase].add_node(speaker)
            graphs[phase].add_node(receiver)
            
            # Add edge with weight (1 for true, 0 for false, skip if NOANNOTATION)
            weight = 1 if sender_label == 'true' else 0 if sender_label == 'false' else None
            if weight is not None:
                if graphs[phase].has_edge(speaker, receiver):
                    graphs[phase][speaker][receiver]['weight'] = (graphs[phase][speaker][receiver]['weight'] + weight) / 2
                    graphs[phase][speaker][receiver]['count'] += 1
                else:
                    graphs[phase].add_edge(speaker, receiver, weight=weight, count=1)
    
    return graphs

def compute_deception_rate(records):
    """Compute deception rate per player per game."""
    deception_rates = defaultdict(lambda: {'true': 0, 'false': 0})
    
    for record in records:
        game_id = record['game_id']
        for speaker, sender_label in zip(record['speakers'], record['sender_labels']):
            if sender_label in ['true', 'false']:
                deception_rates[f"{game_id}_{speaker}"][sender_label] += 1
    
    # Calculate rates
    rates = {}
    for key, counts in deception_rates.items():
        total = counts['true'] + counts['false']
        rates[key] = counts['false'] / total if total > 0 else 0
    
    return rates

def detect_contradictions(records):
    """Detect contradictions in messages sent to different recipients in the same phase."""
    contradictions = []
    
    for record in records:
        game_id = record['game_id']
        phase_messages = defaultdict(list)
        
        # Group messages by phase (year + season)
        for i, (msg, speaker, receiver, season, year) in enumerate(zip(
            record['messages'], record['speakers'], record['receivers'], record['seasons'], record['years']
        )):
            phase = f"{year}_{season}"
            phase_messages[phase].append({
                'message': msg.lower(),
                'speaker': speaker,
                'receiver': receiver,
                'index': i
            })
        
        # Simple contradiction detection: flag if same speaker mentions conflicting moves
        for phase, msgs in phase_messages.items():
            for i, msg1 in enumerate(msgs):
                for j, msg2 in enumerate(msgs[i+1:]):
                    if msg1['speaker'] == msg2['speaker'] and msg1['receiver'] != msg2['receiver']:
                        # Check for conflicting move proposals (basic keyword-based)
                        if ('support' in msg1['message'] and 'attack' in msg2['message']) or \
                           ('attack' in msg1['message'] and 'support' in msg2['message']):
                            contradictions.append({
                                'game_id': game_id,
                                'phase': phase,
                                'speaker': msg1['speaker'],
                                'index1': msg1['index'],
                                'index2': msg2['index'],
                                'score': 1.0  # Placeholder score
                            })
    
    return contradictions

def preprocess_data(file_name, output_dir=OUTPUT_DIR, is_test=False):
    """Preprocess a single JSONL file after filtering empty records."""
    file_path = os.path.join(DATA_DIR, file_name)
    records = load_jsonl(file_path)
    
    # Filter out empty records
    records = filter_empty_records(records)
    if not records:
        print(f"No valid records in {file_name}. Skipping preprocessing.")
        return None, None, None, None, None, None
    
    # Initialize containers
    input_ids = []
    attention_masks = []
    metadata_features = []
    labels = []
    contradiction_scores = []
    deception_rates = []
    game_ids = []
    message_indices = []
    
    # Compute global features
    deception_rates_dict = compute_deception_rate(records)
    contradiction_list = detect_contradictions(records)
    graphs = build_interaction_graph(records)
    
    # Normalize metadata
    metadata_df, season_encoder, year_encoder, players_encoder, game_id_encoder, scaler = normalize_metadata(records)
    
    # Process each record
    metadata_idx = 0
    for idx, record in enumerate(records):
        messages = record['messages']
        sender_labels = record['sender_labels']
        game_id = record['game_id']
        speakers = record['speakers']
        
        # Tokenize messages
        ids, masks = tokenize_messages(messages)
        input_ids.append(ids)
        attention_masks.append(masks)
        
        # Extract metadata for this record
        num_messages = len(messages)
        record_metadata = metadata_df.iloc[metadata_idx:metadata_idx + num_messages]
        metadata_features.append(record_metadata.values)
        metadata_idx += num_messages
        
        # Assign labels (skip for test set if not provided)
        if not is_test:
            labels.append([
                1 if label == 'true' else 0 if label == 'false' else -1
                for label in sender_labels
            ])
        
        # Compute deception rates
        record_deception_rates = [
            deception_rates_dict.get(f"{game_id}_{speaker}", 0.0)
            for speaker in speakers
        ]
        deception_rates.append(record_deception_rates)
        
        # Assign contradiction scores
        record_contradictions = np.zeros(len(messages))
        for contradiction in contradiction_list:
            if contradiction['game_id'] == game_id:
                if contradiction['index1'] in record['absolute_message_index']:
                    local_idx = record['absolute_message_index'].index(contradiction['index1'])
                    record_contradictions[local_idx] = contradiction['score']
                if contradiction['index2'] in record['absolute_message_index']:
                    local_idx = record['absolute_message_index'].index(contradiction['index2'])
                    record_contradictions[local_idx] = contradiction['score']
        contradiction_scores.append(record_contradictions)
        
        # Store game IDs and message indices
        game_ids.extend([game_id] * len(messages))
        message_indices.extend(record['absolute_message_index'])
    
    # Convert to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    metadata_features = torch.tensor(np.concatenate(metadata_features), dtype=torch.float)
    contradiction_scores = torch.tensor(np.concatenate(contradiction_scores), dtype=torch.float)
    deception_rates = torch.tensor(np.concatenate(deception_rates), dtype=torch.float)
    
    if not is_test:
        labels = torch.tensor(np.concatenate(labels), dtype=torch.long)
        data = {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'metadata_features': metadata_features,
            'contradiction_scores': contradiction_scores,
            'deception_rates': deception_rates,
            'labels': labels,
            'game_ids': game_ids,
            'message_indices': message_indices,
            'graphs': graphs
        }
    else:
        data = {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'metadata_features': metadata_features,
            'contradiction_scores': contradiction_scores,
            'deception_rates': deception_rates,
            'game_ids': game_ids,
            'message_indices': message_indices,
            'graphs': graphs
        }
    
    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name.replace('.jsonl', '.pt'))
    torch.save(data, output_path)
    
    return data, season_encoder, year_encoder, players_encoder, game_id_encoder, scaler

def main():
    """Preprocess train, validation, and test datasets."""
    # Process train and validation with labels
    for file_name in ['train.jsonl', 'validation.jsonl']:
        print(f"Preprocessing {file_name}...")
        data, season_encoder, year_encoder, players_encoder, game_id_encoder, scaler = preprocess_data(file_name)
        if data is None:
            print(f"Skipping {file_name} due to no valid records.")
            continue
        print(f"Saved preprocessed data to {os.path.join(OUTPUT_DIR, file_name.replace('.jsonl', '.pt'))}")
        
        # Save encoders and scaler for test set
        if file_name == 'train.jsonl':
            torch.save(season_encoder, os.path.join(OUTPUT_DIR, 'season_encoder.pt'))
            torch.save(year_encoder, os.path.join(OUTPUT_DIR, 'year_encoder.pt'))
            torch.save(players_encoder, os.path.join(OUTPUT_DIR, 'players_encoder.pt'))
            torch.save(game_id_encoder, os.path.join(OUTPUT_DIR, 'game_id_encoder.pt'))
            torch.save(scaler, os.path.join(OUTPUT_DIR, 'scaler.pt'))
    
    # Process test set (no labels)
    print("Preprocessing test.jsonl...")
    data, _, _, _, _, _ = preprocess_data('test.jsonl', is_test=True)
    if data is None:
        print("Skipping test.jsonl due to no valid records.")
    else:
        print(f"Saved preprocessed data to {os.path.join(OUTPUT_DIR, 'test.pt')}")

if __name__ == "__main__":
    main()