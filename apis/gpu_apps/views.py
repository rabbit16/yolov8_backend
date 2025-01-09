import json

from  fastapi import APIRouter, Depends, Response, status, Request

from apis.gpu_apps.schema import PicRequest, FloatArrayModel, RubbishPicRequest, RubbishFloatArrayModel
from utils.person import detect_people, detect_rubbish

query_router = APIRouter(prefix="/gpu_apps", tags=["gpu_apps"])


@query_router.post("/people_monitor", response_model=FloatArrayModel)
def monitor(request: Request, query_request: PicRequest):
    is_d, data, new_people = detect_people(query_request.image_data, query_request.cache_list,
                           query_request.employee_number)
    return FloatArrayModel(data=data, code=0, new_people=new_people)

@query_router.post("/rubbish_monitor", response_model=RubbishFloatArrayModel)
def monitor_rubbish(request: Request, query_request: RubbishPicRequest):
    is_d, data, new_rubbish, new_rubbish_pic = detect_rubbish(query_request.image_data, query_request.cache_list,
                                           query_request.employee_number)
    return RubbishFloatArrayModel(data=data, code=0, new_rubbish=new_rubbish, new_rubbish_pic=new_rubbish_pic)