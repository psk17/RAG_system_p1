from uuid import uuid4

class SessionManager:
    @staticmethod
    def create_session():
        return str(uuid4())
