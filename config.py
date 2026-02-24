import os
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/bank09")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
