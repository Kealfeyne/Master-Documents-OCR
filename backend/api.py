import os

import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return "Все робит на тесте"


# @app.post("/signup/")
# async def signup(request: Request):
#     print("signup...")
#     parsed_data = await request.json()
#     dbresponse = db.signup(parsed_data)
#     return JSONResponse(content=dbresponse, status_code=200)


# @app.post("/login/")
# async def login(request: Request):
#     parsed_data = await request.json()
#     dbresponse = db.login(parsed_data)
#     return JSONResponse(content=dbresponse, status_code=200)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=os.getenv('BACKEND_PORT'))
