from pydantic import BaseModel
from typing import Optional, Tuple, List, Union


class PicRequest(BaseModel):
    """
    该类用来格式化检测投放人数的
    """
    employee_number: str
    image_data: str
    cache_list: str

class FloatArrayModel(BaseModel):
    data: List
    code: int
    new_people: List

class RubbishPicRequest(BaseModel):
    employee_number: str  # 标识是哪个机器发过来的
    image_data: str  # 图像数据
    cache_list: str  # 防止重复检测的数据

class RubbishFloatArrayModel(BaseModel):
    data: List
    code: int
    new_rubbish: List
    new_rubbish_pic: List