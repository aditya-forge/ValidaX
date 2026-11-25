import os
import subprocess
import sys
import time
import re
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 8503
BASE_URL = f"http://localhost:{PORT}"

def main():
    print(f"Starting Streamlit app on port {PORT} in light mode...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["STREAMLIT_THEME_BASE"] = "light"
    creds_dir = ROOT / ".streamlit"
    creds_dir.mkdir(exist_ok=True)
    creds_file = creds_dir / "credentials.toml"
    if not creds_file.exists():
        creds_file.write_text('[general]\nemail = ""\n', encoding="utf-8")
    env["STREAMLIT_CREDENTIALS_FILE"] = str(creds_file)
    
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m", "streamlit", "run",
        str(ROOT / "app" / "app.py"),
        f"--server.port={PORT}",
        "--server.headless=true",
        "--theme.base=light",
        "--browser.gatherUsageStats=false"
    ]
    
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Wait for server to be ready
    health_url = f"{BASE_URL}/_stcore/health"
    ready = False
    for _ in range(30):
        time.sleep(1)
        try:
            res = requests.get(health_url, timeout=1)
            if res.status_code == 200 and res.text.strip() == "ok":
                ready = True
                print("Streamlit app is ready!")
                break
        except Exception:
            pass
            
    if not ready:
        print("Failed to start Streamlit app. Process logs:")
        proc.terminate()
        out, _ = proc.communicate(timeout=5)
        print(out)
        sys.exit(1)
        
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 960},
                device_scale_factor=2,
                color_scheme="light"
            )
            page = context.new_page()
            
            def ensure_light():
                page.evaluate("""() => {
                    const keys = Object.keys(localStorage);
                    for (const k of keys) {
                        if (k.startsWith('stActiveTheme')) localStorage.setItem(k, 'Light');
                    }
                    const btn = document.querySelector('button[data-theme="Light"]');
                    if (btn) btn.click();
                }""")
                time.sleep(1)

            def wait_until_ready():
                # Wait for any spinner / running status to disappear
                try:
                    page.wait_for_selector(".stSpinner", state="detached", timeout=30000)
                except Exception:
                    pass
                for _ in range(40):
                    content = page.content()
                    if "Running load_model_bundle" not in content and "Running load_test_features" not in content:
                        break
                    time.sleep(1)
                time.sleep(2)

            # 1. Overview
            print("Capturing Overview...")
            page.goto(f"{BASE_URL}/", wait_until="networkidle")
            page.wait_for_selector("text=Transaction fraud scoring with reason codes", timeout=30000)
            ensure_light()
            wait_until_ready()
            page.screenshot(path=str(OUTPUT_DIR / "01_overview.png"), full_page=False)
            print("Saved 01_overview.png")

            # 2. Classify (Single)
            print("Capturing Classify (Single)...")
            page.goto(f"{BASE_URL}/classify", wait_until="networkidle")
            page.wait_for_selector("text=Classify", timeout=30000)
            ensure_light()
            wait_until_ready()
            page.screenshot(path=str(OUTPUT_DIR / "02_classify_single.png"), full_page=False)
            print("Saved 02_classify_single.png")

            # 2b. Classify (Batch tab)
            print("Capturing Classify (Batch)...")
            batch_tab = page.get_by_role("tab", name=re.compile("Batch", re.I))
            if batch_tab.count() == 0:
                batch_tab = page.locator('button:has-text("Batch upload")')
            if batch_tab.count() > 0:
                batch_tab.click()
                time.sleep(2)
                wait_until_ready()
                page.screenshot(path=str(OUTPUT_DIR / "02b_classify_batch.png"), full_page=False)
                print("Saved 02b_classify_batch.png")

            # 3. Alerts
            print("Capturing Alerts Queue...")
            page.goto(f"{BASE_URL}/alerts", wait_until="networkidle")
            page.wait_for_selector("text=Alert queue", timeout=30000)
            ensure_light()
            wait_until_ready()
            page.screenshot(path=str(OUTPUT_DIR / "03_alert_queue.png"), full_page=False)
            print("Saved 03_alert_queue.png")

            # 4. Investigate
            print("Capturing Investigation (waiting for full model & SHAP load)...")
            page.goto(f"{BASE_URL}/investigate", wait_until="networkidle")
            ensure_light()
            # Wait for main investigation headers and components to appear
            page.wait_for_selector('text=Transaction details', timeout=60000)
            wait_until_ready()
            time.sleep(3)
            page.screenshot(path=str(OUTPUT_DIR / "04_investigate.png"), full_page=False)
            print("Saved 04_investigate.png")

            # 5. Simulator
            print("Capturing Simulator...")
            page.goto(f"{BASE_URL}/simulator", wait_until="networkidle")
            page.wait_for_selector("text=Threshold simulator", timeout=30000)
            ensure_light()
            wait_until_ready()
            time.sleep(2)
            page.screenshot(path=str(OUTPUT_DIR / "05_simulator.png"), full_page=False)
            print("Saved 05_simulator.png")

            browser.close()
            print("All screenshots successfully captured in light mode!")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
