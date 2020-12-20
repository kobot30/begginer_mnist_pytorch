from enum import Enum


class ModelSelectFlag(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    AUTOENCODER = 3
    GAN = 4  # not use
    CAM = 5
