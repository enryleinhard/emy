from fastapi import Request, APIRouter, UploadFile, File, Response, Form, HTTPException
from fastapi.responses import JSONResponse
from src.emy.emy import DataBase
from src.emy.emy_helpers import get_shipment_data

router = APIRouter(
    prefix="/emy",
    tags=["emy"],
    responses={404: {"description": "Not found"}},
)

@router.get("/supply")
def supply_data():
    try:
        col = DataBase['orders']

        f = {

        }

        p = {
            '_id': 0
        }

        cursor = col.find(filter=f, projection=p, limit=15,)
        supply = list(cursor)

        response = {
            'success': True,
            'data': supply
        }

    except Exception as e:

        response = {
            'success': False,
            'data': []
        }

    return JSONResponse(status_code=200, content=response)

@router.get("/routes")
async def routes():

    try:
        col = DataBase['routes']

        f = {

        }

        p = {
            '_id': 0
        }

        cursor = col.find(filter=f, projection=p)
        routes = list(cursor)

        routes = routes

        response = {
            'success': True,
            'data': routes
        }

    except Exception as e:
        
        print(e)

        response = {
            'success': False,
            'data': []
        }
        
    return JSONResponse(status_code=200, content=response)

@router.post("/route")
async def route(request: Request):

    rjson = await request.json()
        
    try:
        col = DataBase['routes']

        f = {
            'date': rjson['date']
        }

        p = {
            '_id': 0
        }

        cursor = col.find(filter=f, projection=p)
        routes = list(cursor)

        routes = routes[0]['routes'][rjson['driver_number']]

        response = {
            'success': True,
            'data': routes
        }

    except Exception as e:
        
        print(e)

        response = {
            'success': False,
            'data': []
        }
        
    return JSONResponse(status_code=200, content=response)


@router.get("/get-shipment-data")
async def get_data():
    try:
        shipmentData = get_shipment_data()    
        response = {
            'success': True,
            'data': shipmentData
        }

    except Exception as e:
        print('failed')
        print(e)

        response = {
            'success': False,
            'data': []
        }

    return JSONResponse(status_code=200, content=response)


