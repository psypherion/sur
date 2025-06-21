import torch
import torch.nn as nn
from typing import Tuple

class GatedMultimodalRecommender(nn.Module):
    """
    A multimodal recommender system that fuses audio and text features using
    a learned, interpretable gating mechanism, as outlined in the project manifesto.

    This model projects diverse feature vectors into a shared latent space and
    dynamically calculates a gating weight (alpha) to determine the optimal blend
    of audio vs. lyrical modalities for creating a final song embedding.

    Args:
        audio_input_dim (int): The dimensionality of the input audio feature vector
                               (e.g., scalars + harmonics + MFCCs).
        text_input_dim (int): The dimensionality of the input text feature vector
                              (e.g., semantic embedding + sentiment scores).
        embedding_dim (int): The target dimension for the shared latent space. This
                             is a key hyperparameter.
        gate_hidden_dim (int): The size of the hidden layer within the gating network.
    """
    def __init__(self, audio_input_dim: int, text_input_dim: int, embedding_dim: int = 128, gate_hidden_dim: int = 64):
        super().__init__()
        
        self.audio_input_dim = audio_input_dim
        self.text_input_dim = text_input_dim
        self.embedding_dim = embedding_dim
        self.gate_hidden_dim = gate_hidden_dim

        # --- 1. Projection Layers ---
        # Projects each modality into the common embedding space. This crucial step
        # solves the problem of mismatched input vector sizes and allows the model
        # to learn an optimal "translation" for each modality.
        self.audio_projector = nn.Linear(self.audio_input_dim, self.embedding_dim)
        self.text_projector = nn.Linear(self.text_input_dim, self.embedding_dim)
        
        # A non-linear activation function after projection allows for more complex mappings.
        self.projection_activation = nn.ReLU()

        # --- 2. The Gating Network (The Core Novelty) ---
        # This sub-network is the heart of the explainability. It learns the dynamic
        # weight 'alpha' by looking at the projected representations of both modalities.
        self.gating_network = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout adds regularization to prevent overfitting.
            nn.Linear(self.gate_hidden_dim, 1),
            nn.Sigmoid()      # Sigmoid activation constrains alpha to a value between 0 and 1.
        )

    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass of the model, implementing the multimodal fusion.

        Args:
            audio_features (torch.Tensor): A batch of audio feature vectors.
                                           Shape: (batch_size, audio_input_dim)
            text_features (torch.Tensor): A batch of text feature vectors.
                                          Shape: (batch_size, text_input_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - final_embedding: The fused, L2-normalized song embeddings.
                                   Shape: (batch_size, embedding_dim)
                - alpha: The learned gating weights for explainability.
                         Shape: (batch_size, 1)
        """
        # Step 1: Project raw features into the shared embedding space.
        projected_audio = self.projection_activation(self.audio_projector(audio_features))
        projected_text = self.projection_activation(self.text_projector(text_features))

        # Step 2: Calculate the gating weight 'alpha'.
        concatenated_features = torch.cat((projected_audio, projected_text), dim=1)
        alpha = self.gating_network(concatenated_features)

        # Step 3: Fuse the modalities using the learned gate.
        # This weighted sum is the implementation of our core hypothesis.
        fused_embedding = (alpha * projected_audio) + ((1 - alpha) * projected_text)
        
        # Step 4: L2 Normalize the final embedding (critical for metric learning).
        # This ensures embeddings lie on a unit hypersphere, making cosine similarity
        # a stable and meaningful distance metric.
        final_embedding = nn.functional.normalize(fused_embedding, p=2, dim=1)
        
        return final_embedding, alpha

# --- Self-Testing Block ---
if __name__ == '__main__':
    # This block runs only when the script is executed directly (e.g., `python model.py`)
    # It serves as a unit test to verify the model's integrity.
    
    print("--- Running a self-test of the GatedMultimodalRecommender model ---")

    # Use the exact dimensions from your preprocessor.py output
    AUDIO_DIM = 174
    TEXT_DIM = 387
    EMBEDDING_DIM = 128
    BATCH_SIZE = 4 # A small batch size for the test

    # Instantiate the model
    model = GatedMultimodalRecommender(
        audio_input_dim=AUDIO_DIM,
        text_input_dim=TEXT_DIM,
        embedding_dim=EMBEDDING_DIM
    )
    
    print("\nModel Architecture:")
    print(model)

    # Create dummy input tensors
    dummy_audio_input = torch.randn(BATCH_SIZE, AUDIO_DIM)
    dummy_text_input = torch.randn(BATCH_SIZE, TEXT_DIM)

    # Perform a forward pass in evaluation mode
    model.eval()
    with torch.no_grad():
        final_song_vectors, gating_weights = model(dummy_audio_input, dummy_text_input)

    # --- Verification Checks ---
    print(f"\n--- Test Results (Batch Size = {BATCH_SIZE}) ---")
    
    # Check output shapes
    print(f"Shape of Final Song Vectors: {final_song_vectors.shape}")
    print(f"Expected Final Shape:        ({BATCH_SIZE}, {EMBEDDING_DIM})")
    assert final_song_vectors.shape == (BATCH_SIZE, EMBEDDING_DIM), "Final embedding shape is incorrect!"
    
    print(f"\nShape of Gating Weights (Alpha): {gating_weights.shape}")
    print(f"Expected Alpha Shape:            ({BATCH_SIZE}, 1)")
    assert gating_weights.shape == (BATCH_SIZE, 1), "Gating weight shape is incorrect!"

    # Check normalization
    norms = torch.norm(final_song_vectors, p=2, dim=1)
    print(f"\nNorms of output vectors (should all be ~1.0): {norms.numpy()}")
    assert torch.allclose(norms, torch.ones(BATCH_SIZE)), "L2 Normalization failed!"

    # Check alpha values
    print(f"Example Alpha values (should be between 0 and 1): {gating_weights.squeeze().numpy()}")
    assert torch.all(gating_weights >= 0) and torch.all(gating_weights <= 1), "Alpha values are out of the [0, 1] range!"

    print("\n✅ Model self-test passed successfully!")