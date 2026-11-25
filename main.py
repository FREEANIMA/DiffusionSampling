# main.py (프로젝트 루트)
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        app_dir="_backend",          # 이 폴더를 sys.path에 추가
        host="127.0.0.1",
        port=8000,
        reload=True,
    )