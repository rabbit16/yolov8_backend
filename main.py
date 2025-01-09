import json

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from apis.gpu_apps import views as gpu_views
app = FastAPI(
    debug=True,
)

# 解决跨域问题
orgins = [
    "*"
]

app.include_router(gpu_views.query_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=orgins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=9999)
    # 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