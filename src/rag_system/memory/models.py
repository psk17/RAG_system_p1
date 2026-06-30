from datetime import datetime
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime
