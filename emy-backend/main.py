import uvicorn  
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from routes.api import router as api_router
from imp import reload

app = FastAPI(
    title = "EMY-API",
    description="Backend API for Emy.",
    version = "1.0.0",
    docs_url = '/api',
    openapi_url = '/api/openapi/json',
    redoc_url = None
)

# origins = [
#     "http://localhost",
#     "http://localhost:8080",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=9090, log_level="info", reload=True)
    print("running")
