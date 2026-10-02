"""
Task 2: Specialist autoencoders for hard-routed restoration.
Each specialist reuses the Task 1 UniversalAutoencoder architecture but
is trained ONLY on one corruption type, with independent parameters.
"""
from src.models.task1_autoencoder import UniversalAutoencoder


class SpecialistAutoencoder(UniversalAutoencoder):
    """Identical architecture to Task 1's model; kept as a distinct class
    name for clarity in checkpoints/configs, since each specialist has
    independently trained weights despite sharing the same architecture."""
    pass
