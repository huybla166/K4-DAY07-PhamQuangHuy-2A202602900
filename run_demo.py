import sys
import webbrowser
import threading
import time
import uvicorn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("====================================================================")
    print(">>> DANG KHOI DONG LAB 07 RAG DEMO WEB SERVER...")
    print(">>> Dia chi truy cap: http://127.0.0.1:8000")
    print("Nhan Ctrl+C de dung may chu bat cu luc nao.")
    print("====================================================================")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, log_level="info")
