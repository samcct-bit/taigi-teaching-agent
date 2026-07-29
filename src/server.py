# 本地語音 API 服務器 (server.py)
import os
import sys
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 將當前目錄與父目錄加入搜尋路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from stt.generator import TaigiSTT

app = FastAPI(title="Taigi ASR Local Server")

# 以專案根目錄的 config.json 為準（避免以 src 為工作目錄啟動時讀不到設定而默默降級為 dummy）
project_root = os.path.dirname(current_dir)
config_path = os.path.join(project_root, "config.json")

# 啟動時建立單一 STT 實例並重用，避免每次請求重新載入 whisper 模型（耗時且吃記憶體）
stt = TaigiSTT(config_path)

# 設定前端儲存自訂教材的目錄路徑並掛載為靜態檔案服務 (供雲端網頁讀取剛產生的本機檔案)
FRONTEND_DIR = r"d:\antigravity\taigi-class"
FRONTEND_CUSTOM_DIR = os.path.join(FRONTEND_DIR, "data", "custom")
os.makedirs(FRONTEND_CUSTOM_DIR, exist_ok=True)
app.mount("/local_custom", StaticFiles(directory=FRONTEND_CUSTOM_DIR), name="local_custom")

# 設定 CORS 中間件，允許離線網頁 (file:/// 協議) 跨網域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Taigi ASR Local Server is running. Ready for speech-to-text requests."
    }

@app.post("/api/stt")
async def speech_to_text_endpoint(
    file: UploadFile = File(...),
    target_text: str = Form("")
):
    """
    語音轉文字 API 接口
    - file: 瀏覽器錄音上傳的音訊檔案 (一般為 audio/webm 或 audio/wav)
    - target_text: 預期的台語漢字，用於 Dummy ASR 模擬比對
    """
    # 決定副檔名並保存為臨時檔案
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".webm"
    if not suffix:
        suffix = ".webm"
        
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}{suffix}")
    
    try:
        # 寫入上傳的音訊資料
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # 使用啟動時建立的全域 STT 實例執行辨識（模型已快取）
        recognized_text = stt.speech_to_text(temp_path, target_text)
        return {"text": recognized_text}
        
    except Exception as e:
        print(f"[-] 伺服器處理 ASR 發生異常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 清除臨時檔案
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/generate")
async def generate_lesson(
    prompt: str = Form(...),
    grade: str = Form("國中七年級"),
    skip_media: bool = Form(False),
    include_video: bool = Form(False)
):
    """
    自訂教材一鍵生成 API 接口
    - prompt: 教學主題或自然語言描述
    - grade: 適用年級 (預設: 國中七年級)
    - skip_media: 是否跳過音訊/圖片生成（用於測試，預設 False）
    - include_video: 是否同步生成教學影片（預設 False）
    """
    try:
        from agent.natural_language_runner import run_natural_language_request, _safe_folder_name, parse_natural_language_request
        import time
        import tempfile
        import shutil
        import base64
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # 解析主題與適用年級
        parsed = parse_natural_language_request(prompt, default_grade=grade)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_topic = _safe_folder_name(parsed["topic"])
        folder_name = f"custom_{timestamp}_{safe_topic}"
        
        # 使用系統暫存資料夾，避免雲端環境跨資料夾權限問題
        temp_dir = tempfile.mkdtemp()
        local_output_dir = os.path.join(temp_dir, folder_name)
        os.makedirs(local_output_dir, exist_ok=True)
        
        # 呼叫自然語言教材生成鏈
        manifest = run_natural_language_request(
            request=prompt,
            grade=grade,
            config_path=config_path,
            output_dir=local_output_dir,
            include_video=include_video,
            skip_media=skip_media,
            validate_output=False  # 跳過驗證以加速生成
        )
        
        # GitHub 推送邏輯 (若有設定 PAT 則推送到 GitHub，否則存回本機)
        github_pat = os.getenv("GITHUB_PAT")
        repo_name = os.getenv("GITHUB_REPO", "samcct-bit/taigi-class")
        
        if github_pat:
            print(f"[*] 正在上傳生成的教材到 GitHub: {repo_name}...")
            from github import Github, InputGitTreeElement
            g = Github(github_pat)
            repo = g.get_repo(repo_name)
            
            commit_message = f"feat(ai): add generated lesson '{safe_topic}'"
            master_ref = repo.get_git_ref("heads/master")
            master_sha = master_ref.object.sha
            base_tree = repo.get_git_tree(master_sha)
            
            tree_elements = []
            for root, dirs, files in os.walk(local_output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = f"data/custom/{folder_name}/" + os.path.relpath(file_path, local_output_dir).replace("\\", "/")
                    
                    with open(file_path, "rb") as f:
                        content = f.read()
                    
                    if file_path.endswith((".wav", ".mp3", ".ogg", ".mp4", ".png", ".jpg")):
                        # 二進位檔案：需透過 base64 上傳到 Blob
                        encoded = base64.b64encode(content).decode("utf-8")
                        blob = repo.create_git_blob(encoded, "base64")
                        tree_elements.append(InputGitTreeElement(path=rel_path, mode="100644", type="blob", sha=blob.sha))
                    else:
                        blob = repo.create_git_blob(content.decode("utf-8"), "utf-8")
                        tree_elements.append(InputGitTreeElement(path=rel_path, mode="100644", type="blob", sha=blob.sha))
            
            new_tree = repo.create_git_tree(tree_elements, base_tree)
            new_commit = repo.create_git_commit(commit_message, new_tree, [repo.get_git_commit(master_sha)])
            master_ref.edit(new_commit.sha)
            print(f"[+] 成功上傳到 GitHub！Commit: {new_commit.sha}")
        else:
            frontend_custom_dir = os.path.join(FRONTEND_CUSTOM_DIR, folder_name)
            shutil.copytree(local_output_dir, frontend_custom_dir, dirs_exist_ok=True)
            print("[*] 無 GITHUB_PAT，已退回儲存於本機資料夾。")
            
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "status": "success",
            "folderName": folder_name,
            "title": parsed["topic"]
        }
        
    except Exception as e:
        print(f"[-] 伺服器處理自訂教材生成發生異常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 啟動本地伺服器，運行於 127.0.0.1:8000
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
