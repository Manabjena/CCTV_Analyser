import uuid
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from datetime import datetime

class CameraID(str, Enum):
    CAM_1 = "CAM_1"
    CAM_2 = "CAM_2"
    CAM_3 = "CAM_3"
    CAM_4 = "CAM_4"
    CAM_5 = "CAM_5"

class RetailEventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    QUEUE_ABANDONMENT = "QUEUE_ABANDONMENT"
    SESSION_END = "SESSION_END"
    STAFF_BACKROOM_LOG = "STAFF_BACKROOM_LOG"

class StoreZoneID(str, Enum):
    ENTRANCE_ZONE = "ENTRANCE_ZONE"
    BILLING_ZONE = "BILLING_ZONE"
    STORAGE_ROOM = "STORAGE_ROOM"
    MAKEUP = "MAKEUP"
    FRAGRANCE = "FRAGRANCE"
    SKIN = "SKIN"
    HAIR = "HAIR"
    MAIN_FLOOR = "MAIN_FLOOR"

class EventMetadataPayload(BaseModel):
    queue_depth: Optional[int] = Field(default=None, ge=0, le=100)
    sku_zone: Optional[StoreZoneID] = Field(default=None)
    session_seq: int = Field(..., ge=1)
    frame_index: int = Field(..., ge=0)
    video_time_seconds: float = Field(..., ge=0.0)

class StoreTrackingEvent(BaseModel):
    event_id: uuid.UUID
    store_id: str = Field(default="ST1008")
    camera_id: CameraID
    visitor_id: str = Field(..., min_length=4, max_length=50)
    event_type: RetailEventType
    timestamp: str
    zone_id: Optional[StoreZoneID] = Field(default=None)
    dwell_ms: int = Field(default=0, ge=0)
    is_staff: bool = Field(default=False)
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: EventMetadataPayload

    @field_validator("timestamp")
    @classmethod
    def validate_iso_timestamp(cls, timestamp_str: str) -> str:
        """
        Validates that string values conform completely to standard ISO-8601 formatting specs.

        Inputs:
            timestamp_str (str): Raw timestamp string delivered from network buffers.

        Outputs:
            str: Verified timestamp matching standard system structures.
        """
        try:
            cleaned_str = timestamp_str.replace("Z", "+00:00")
            datetime.fromisoformat(cleaned_str)
        except ValueError:
            raise ValueError("Timestamp value does not conform to strict ISO-8601 formatting rules.")
        return timestamp_str