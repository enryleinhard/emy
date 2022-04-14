from fastapi import Request, APIRouter, UploadFile, File, Response, Form, HTTPException
from fastapi.responses import JSONResponse
from src.emy.emy import DataBase

router = APIRouter(
    prefix="/tracker",
    tags=["tracker"],
    responses={404: {"description": "Not found"}},
)

@router.post("/add-route")
async def add_route(request: Request):
    
    rjson = await request.json()

    print(rjson)

    try:
        col = DataBase['tracker']

        rjson['date'] = '4/5/2022'

        col.insert_one(rjson)
        
        response = {
            'success': False,
            'message': 'Succesfully added'
        }

    except Exception as e:
        
        print(e)

        response = {
            'success': False,
            'data': []
        }

    return response

@router.post("/route")
async def get_real_route(request: Request):
    
    rjson = await request.json()

    try:
        col = DataBase['tracker']
        f = {
            'date' : rjson['date']
        }
        p = {
            '_id': 0
        }
        s = [('driver', 1)]
        cursor = col.find(filter=f,projection=p, sort=s)
        routes = list(cursor)

        new_routes = []

        for x in routes:
            new_routes.append(x['tracked'])

        response = {
            'success': True,
            'data': new_routes
        }

    except Exception as e:
        
        print(e)

        response = {
            'success': False,
            'data': []
        }

    return response