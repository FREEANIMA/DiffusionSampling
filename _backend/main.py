# main.py (프로젝트 루트, api 폴더와 같은 위치)
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)