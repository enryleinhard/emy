from fastapi import APIRouter
from src.optimizer import optimizer_api
from src.emy import emy_api
from src.tracker import tracker_api

router = APIRouter()

router.include_router(optimizer_api.router)
router.include_router(emy_api.router)
router.include_router(tracker_api.router)
