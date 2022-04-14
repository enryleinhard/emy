from fastapi import Request, APIRouter, UploadFile, File, Response, Form, HTTPException
from fastapi.responses import JSONResponse
from src.optimizer.optimizer import DataBase
from src.optimizer.optimizer import optimize_expected

router = APIRouter(
    prefix="/route-optimizer",
    tags=["optimizer"],
    responses={404: {"description": "Not found"}},
)

@router.post("/optimize-route")
def optimize_route(dropoff_date: str = Form(...)):
    optimizedRoute = optimize_expected(dropoff_date)
    DataBase.routes.delete_one({"date":dropoff_date})
    DataBase.routes.insert_one(optimizedRoute)
    return JSONResponse(status_code=200, content={"message": f"All Success for {dropoff_date}"})


@router.post("/update-route")
async def update_route(request: Request):
    
    rjson = await request.json()

    print(rjson)

    try:
        col = DataBase['routes']

        f = {
            'date': rjson['date']
        }

        pu = {
            '_id': 1
        }

        cursor = col.find(filter=f, projection=pu)
        routes = list(cursor)[0]

        uvalues = {
            "$set": {
                'real_distance_traveled': rjson["real_distance_traveled"],
                'total_CO2_emission': rjson["total_CO2_emission"],
            }
        }
        fu = {
            '_id': routes['_id']
        }

        col.update_one(fu, uvalues)

        p = {
            '_id': 0
        }

        cursor = col.find(filter=f, projection=p)
        routes = list(cursor)[0]

        # print(routes)

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

    return response