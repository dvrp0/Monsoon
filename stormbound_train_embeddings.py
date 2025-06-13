import json
import pickle
import re
from typing import Dict, List, Any
import argparse
from pathlib import Path

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import StandardScaler
import torch


def clean_text(text: str) -> str:
    """Clean and normalize card text"""
    if not text:
        return ""

    # Remove special characters and normalize
    text = re.sub(r'[^\w\s가-힣]', ' ', text)  # Keep Korean characters
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def extract_card_features(card: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features from a Stormbound card"""
    features = {}

    # Basic attributes
    features['cost'] = card.get('cost', 0)
    features['type'] = card.get('type', '')
    features['faction'] = card.get('kingdom', '')
    features['rarity'] = card.get('rarity', '')

    # Strength (for units/structures)
    strengths = card.get('strengths', [])
    if strengths:
        features['base_strength'] = strengths[0] if len(strengths) > 0 else 0
        features['max_strength'] = strengths[-1] if len(strengths) > 0 else 0
    else:
        features['base_strength'] = 0
        features['max_strength'] = 0

    # Movement (for units)
    features['movement'] = card.get('movement', 0)

    # Unit types
    unit_types = card.get('unitTypes', '')
    features['unit_type'] = unit_types if unit_types else ''

    # Abilities
    ability = card.get('ability', {})
    features['has_ability'] = len(ability) > 0

    # Extract ability values if present
    for key, values in ability.items():
        if isinstance(values, list) and values:
            features[f'ability_{key}_base'] = values[0]
            features[f'ability_{key}_max'] = values[-1]

    # Card text for semantic features
    description = card.get('description', '')
    features['description'] = clean_text(description)

    return features


def create_card_documents(cards_data: List[Dict[str, Any]]) -> List[TaggedDocument]:
    """Create TaggedDocument objects for Doc2Vec training"""
    documents = []

    for card in cards_data:
        card_id = card['id']
        features = extract_card_features(card)

        # Create text representation
        text_parts = []

        # Add card type and faction
        if features['type']:
            text_parts.append(features['type'])
        if features['faction']:
            text_parts.append(features['faction'])
        if features['rarity']:
            text_parts.append(features['rarity'])
        if features['unit_type']:
            text_parts.append(features['unit_type'])

        # Add cost information
        cost = features['cost']
        if cost <= 2:
            text_parts.append('저비용')
        elif cost <= 4:
            text_parts.append('중비용')
        else:
            text_parts.append('고비용')

        # Add strength information
        base_str = features['base_strength']
        if base_str > 0:
            if base_str <= 3:
                text_parts.append('약한유닛')
            elif base_str <= 6:
                text_parts.append('보통유닛')
            else:
                text_parts.append('강한유닛')

        # Add movement information
        movement = features['movement']
        if movement == 0:
            text_parts.append('이동불가')
        elif movement == 1:
            text_parts.append('이동1')
        else:
            text_parts.append('고이동성')

        # Add ability flag
        if features['has_ability']:
            text_parts.append('능력보유')

        # Add description if available
        if features['description']:
            # Split description into words
            desc_words = features['description'].split()
            text_parts.extend(desc_words)

        # Create document
        doc = TaggedDocument(words=text_parts, tags=[card_id])
        documents.append(doc)

    return documents


def create_numerical_features(cards_data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Create numerical feature vectors for cards"""
    card_features = {}

    # Collect all ability keys
    all_ability_keys = set()
    for card in cards_data:
        ability = card.get('ability', {})
        for key in ability.keys():
            all_ability_keys.add(key)

    all_ability_keys = sorted(list(all_ability_keys))

    for card in cards_data:
        card_id = card['id']
        features = extract_card_features(card)

        # Create feature vector
        feature_vector = []

        # Basic numerical features
        feature_vector.append(features['cost'])
        feature_vector.append(features['base_strength'])
        feature_vector.append(features['max_strength'])
        feature_vector.append(features['movement'])
        feature_vector.append(1 if features['has_ability'] else 0)

        # One-hot encode type
        type_options = ['유닛', '건물', '주문']
        for type_opt in type_options:
            feature_vector.append(1 if features['type'] == type_opt else 0)

        # One-hot encode faction
        faction_options = ['중립', '겨울', '아이언클래드', '섀도우펜', '스웜']
        for faction_opt in faction_options:
            feature_vector.append(1 if features['faction'] == faction_opt else 0)

        # One-hot encode rarity
        rarity_options = ['일반', '레어', '에픽', '전설']
        for rarity_opt in rarity_options:
            feature_vector.append(1 if features['rarity'] == rarity_opt else 0)

        # Ability features
        for ability_key in all_ability_keys:
            has_ability = f'ability_{ability_key}_base' in features
            feature_vector.append(1 if has_ability else 0)
            if has_ability:
                feature_vector.append(features[f'ability_{ability_key}_base'])
                feature_vector.append(features[f'ability_{ability_key}_max'])
            else:
                feature_vector.append(0)
                feature_vector.append(0)

        card_features[card_id] = np.array(feature_vector, dtype=np.float32)

    return card_features


def train_doc2vec_model(documents: List[TaggedDocument], vector_size: int = 64, epochs: int = 100):
    """Train Doc2Vec model on card documents"""
    print(f"Training Doc2Vec model with {len(documents)} documents...")

    model = Doc2Vec(
        documents,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        epochs=epochs,
        alpha=0.025,
        min_alpha=0.00025,
        dm=1  # PV-DM
    )

    print("Doc2Vec training completed!")
    return model


def combine_features(doc2vec_model: Doc2Vec, numerical_features: Dict[str, np.ndarray], 
                    cards_data: List[Dict[str, Any]], target_dim: int = 128) -> Dict[str, np.ndarray]:
    """Combine Doc2Vec and numerical features into final embeddings"""
    print("Combining features...")

    # Get all numerical feature vectors
    num_features_list = []
    card_ids = []

    for card in cards_data:
        card_id = card['id']
        if card_id in numerical_features:
            num_features_list.append(numerical_features[card_id])
            card_ids.append(card_id)

    if not num_features_list:
        raise ValueError("No numerical features found!")

    # Normalize numerical features
    num_features_array = np.vstack(num_features_list)
    scaler = StandardScaler()
    num_features_normalized = scaler.fit_transform(num_features_array)

    # Get semantic features from Doc2Vec
    semantic_features = []
    for card_id in card_ids:
        try:
            semantic_vec = doc2vec_model.dv[card_id]
            semantic_features.append(semantic_vec)
        except KeyError:
            print(f"Warning: Card {card_id} not found in Doc2Vec model")
            semantic_features.append(np.zeros(doc2vec_model.vector_size))

    semantic_features_array = np.vstack(semantic_features)

    # Combine features
    combined_features = np.hstack([num_features_normalized, semantic_features_array])

    # Reduce to target dimension if necessary
    if combined_features.shape[1] > target_dim:
        # Use PCA-like dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim)
        combined_features = pca.fit_transform(combined_features)
        print(f"Reduced from {combined_features.shape[1]} to {target_dim} dimensions using PCA")
    elif combined_features.shape[1] < target_dim:
        # Pad with zeros
        padding = np.zeros((combined_features.shape[0], target_dim - combined_features.shape[1]))
        combined_features = np.hstack([combined_features, padding])
        print(f"Padded from {combined_features.shape[1]} to {target_dim} dimensions")

    # Create final embeddings dictionary
    final_embeddings = {}
    for i, card_id in enumerate(card_ids):
        final_embeddings[card_id] = combined_features[i]

    print(f"Created embeddings for {len(final_embeddings)} cards with dimension {target_dim}")
    return final_embeddings


def main():
    parser = argparse.ArgumentParser(description="Train Stormbound card embeddings")
    parser.add_argument("--cards_json", required=True, help="Path to cards.json file")
    parser.add_argument("--output_path", default="stormbound_card_embeddings.pkl", 
                       help="Output path for embeddings pickle file")
    parser.add_argument("--doc2vec_output", default="stormbound_doc2vec_model", 
                       help="Output path for Doc2Vec model")
    parser.add_argument("--vector_size", type=int, default=64, 
                       help="Doc2Vec vector size")
    parser.add_argument("--target_dim", type=int, default=128, 
                       help="Final embedding dimension")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Doc2Vec training epochs")

    args = parser.parse_args()

    # Load cards data
    print(f"Loading cards from {args.cards_json}")
    with open(args.cards_json, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)

    print(f"Loaded {len(cards_data)} cards")

    # Create documents for Doc2Vec
    documents = create_card_documents(cards_data)
    print(f"Created {len(documents)} documents for Doc2Vec training")

    # Train Doc2Vec model
    doc2vec_model = train_doc2vec_model(documents, args.vector_size, args.epochs)

    # Save Doc2Vec model
    doc2vec_model.save(args.doc2vec_output)
    print(f"Saved Doc2Vec model to {args.doc2vec_output}")

    # Create numerical features
    numerical_features = create_numerical_features(cards_data)
    print(f"Created numerical features for {len(numerical_features)} cards")

    # Combine features
    final_embeddings = combine_features(doc2vec_model, numerical_features, cards_data, args.target_dim)

    # Save embeddings
    with open(args.output_path, 'wb') as f:
        pickle.dump(final_embeddings, f)

    print(f"Saved final embeddings to {args.output_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 