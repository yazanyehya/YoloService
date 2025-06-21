import boto3
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from boto3.dynamodb.conditions import Key
from storage.base import PredictionStorage
from decimal import Decimal



class DynamoDBStorage(PredictionStorage):
    def __init__(self, region="us-west-1"):
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.sessions_table = self.dynamodb.Table("yazan-dev-prediction_sessions")
        self.objects_table = self.dynamodb.Table("yazan-dev-detection_objects")

    def save_prediction(self, uid: str, original_image: str, predicted_image: str) -> None:
        response = self.sessions_table.put_item(Item={
            "uid": uid,
            "timestamp": datetime.utcnow().isoformat(),
            "original_image": original_image,
            "predicted_image": predicted_image
        })
        print("📥 DynamoDB put_item response:", response)
        print("UID: ",uid)

    def save_detection(self, uid, label, score, bbox):
        print(f"🔸 Saving detection for uid={uid} | label={label}, score={score}, bbox={bbox}")

        item = {
            "id": str(uuid.uuid4()),  # unique ID for each detection
            "prediction_uid": uid,  # this is required for querying
            "label": label,
            "score": Decimal(str(score)),
            "bbox": bbox
        }

        print(f"🟡 DynamoDB item to save: {item}")
        self.objects_table.put_item(Item=item)

    def get_prediction(self, uid: str) -> Optional[Dict]:
        print(f"🔍 Fetching prediction session with uid={uid}")
        response = self.sessions_table.get_item(Key={"uid": uid})
        session = response.get("Item")
        if not session:
            print("❌ No session found.")
            return None

        print("✅ Session found:", session)

        try:
            objects_response = self.objects_table.query(
                IndexName="prediction_uid-index",
                KeyConditionExpression=Key("prediction_uid").eq(uid)
            )
            objects = objects_response.get("Items", [])
            print(f"🟢 Found {len(objects)} detection objects for uid={uid}")
        except Exception as e:
            print(f"❌ Failed to query detection objects: {e}")
            objects = []

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
