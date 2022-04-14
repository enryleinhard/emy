from src.emy.emy import DataBase
def get_truck_data(truckRoute):
    start = truckRoute["routes"][0]["orig_store"]
    end = truckRoute["routes"][-1]["dest_store"]
    emission = truckRoute["emission"]
    weight = truckRoute["weight"]

    output = { 
             'Start': start,
             'End' : end,
             "Emission" : emission,
             "Weight" : weight
    }

    return output

def get_shipment_data():
    shipmentData = DataBase.routes.find_one({})
    output = {
        "shipmentId":"SH21",
        "date": '4/5/2022',
        "truck": {}
    }
    truckCount = 1
    truckDict = {}
    for truckRoute in shipmentData["real_distance_traveled"]:
        truckId = f"A{truckCount}"

        start = truckRoute["routes"][0]["orig_store"]
        end = truckRoute["routes"][-1]["dest_store"]
        emission = truckRoute["emission"]
        weight = truckRoute["weight"]

        truckOutput = { 
                'Start': start,
                'End' : end,
                "Emission" : emission,
                "Weight" : weight
        }

        truckDict[truckId] = truckOutput
        truckCount += 1
    output["truck"] = truckDict
    return output