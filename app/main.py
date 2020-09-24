from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import predict, viz, vizmap, vizprices

app = FastAPI(
    title='AirBnB DS API',
    description='',
    version='0.1',
    docs_url='/',
)

app.include_router(predict.router)
app.include_router(viz.router)
app.include_router(vizmap.router)
app.include_router(vizprices.router)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex='https?://.*',
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
