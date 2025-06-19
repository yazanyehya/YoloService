import boto3
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from boto3.dynamodb.conditions import Key
from storage.base import PredictionStorage


class DynamoDBStorage(PredictionStorage):
    def __init__(self, region="us-west-1"):
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.sessions_table = self.dynamodb.Table("yazan-dev-prediction_sessions")
        self.objects_table = self.dynamodb.Table("yazan-dev-detection_objects")

    def save_prediction(self, uid: str, original_image: str, predicted_image: str) -> None:
        self.sessions_table.put_item(Item={
            "uid": uid,
            "timestamp": datetime.utcnow().isoformat(),
            "original_image": original_image,
            "predicted_image": predicted_image
        })

    def save_detection(self, prediction_uid: str, label: str, score: float, box: str) -> None:
        self.objects_table.put_item(Item={
            "id": str(uuid.uuid4()),
            "prediction_uid": prediction_uid,
            "label": label,
            "score": score,
            "box": box
        })

    def get_prediction(self, uid: str) -> Optional[Dict]:
        response = self.sessions_table.get_item(Key={"uid": uid})
        session = response.get("Item")
        if not session:
            return None

        objects = self.objects_table.query(
            IndexName="prediction_uid-index",
            KeyConditionExpression=Key("prediction_uid").eq(uid)
        )["Items"]

        session["detection_objects"] = objects
        return session

    def get_predictions_by_label(self, label: str) -> List[Dict]:
        response = self.objects_table.query(
            IndexName="label-index",
            KeyConditionExpression=Key("label").eq(label)
        )
        uids = list({item["prediction_uid"] for item in response.get("Items", [])})
        return [{"uid": uid} for uid in uids]

    def get_predictions_by_score(self, min_score: float) -> List[Dict]:
        # DynamoDB does not support range filtering unless indexed, so we scan
        scan = self.objects_table.scan()
        filtered = [
            obj for obj in scan.get("Items", [])
            if float(obj.get("score", 0)) >= min_score
        ]
        uids = list({item["prediction_uid"] for item in filtered})
        return [{"uid": uid} for uid in uids]
