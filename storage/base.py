from abc import ABC, abstractmethod
from typing import List, Dict

class PredictionStorage(ABC):
    @abstractmethod
    def save_prediction(self, uid: str, original_image: str, predicted_image: str) -> None:
        pass

    @abstractmethod
    def save_detection(self, prediction_uid: str, label: str, score: float, box: str) -> None:
        pass

    @abstractmethod
    def get_prediction(self, uid: str) -> Dict:
        pass

    @abstractmethod
    def get_predictions_by_label(self, label: str) -> List[Dict]:
        pass

    @abstractmethod
    def get_predictions_by_score(self, min_score: float) -> List[Dict]:
        pass
