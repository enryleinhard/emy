import pymongo
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


SECURITY_STRING = "mongodb+srv://anagata:adminanagata@cluster0.penfp.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
client = pymongo.MongoClient(SECURITY_STRING)
DataBase = client['proddb']