from abc import ABC, abstractmethod
import numpy as np

class BaseSpeechSeparationModel(ABC):
    """
    Abstract base class for all speech separation models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self, checkpoint_path: str):
        pass

    @abstractmethod
    def preprocess(self, audio_path: str, sample_rate: int):
        pass

    @abstractmethod
    def separate(self, processed_audio):
        pass

    @abstractmethod
    def postprocess(self, outputs):
        pass

    @abstractmethod
    def unload_model(self):
        """
        Free model from memory (CPU/GPU).
        """
        pass

    def infer(self, audio_path:str):
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded!")

        x = self.preprocess(audio_path)
        outputs = self.separate(x)
        results = self.postprocess(outputs)

        return results