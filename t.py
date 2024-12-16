import certifi
from pymongo import MongoClient
from pymongo.server_api import ServerApi

client = MongoClient(
    host="mongodb+srv://mrskazakova13:eductionit13_@cluster0.unxea.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    server_api=ServerApi("1"),
    tlsAllowInvalidCertificates=True
)

databases = client.list_database_names()
print("Доступные базы данных:", databases)