# main.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types
from datetime import datetime
import json
import os  # เพิ่มเข้ามาเพื่อดึงข้อมูลระบบ Environment Variable

from data.birthday_data import BIRTHDAY_FLOWERS

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ดึงค่า API Key จากระบบ Environment Variable ของ Render เพื่อความปลอดภัยสูงสุด
api_key_env = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key_env)


def get_flower_for_date(month: str, day: str) -> str:
    month_data = BIRTHDAY_FLOWERS.get(month, {})
    return month_data.get(day, "The Seasonal Heritage Blossom")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    now = datetime.now()
    current_month_name = now.strftime("%B") 
    current_day_str = str(now.day)          
    
    today_flower = get_flower_for_date(current_month_name, current_day_str)
    today_date_display = f"{current_month_name} {current_day_str}"
    
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "today_date": today_date_display,
        "today_flower": today_flower,
        "local_flowers_json": json.dumps(BIRTHDAY_FLOWERS)
    })

@app.post("/identify")
async def identify_flower(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    recovery_prompt = (
        "Use the image to provide concise plant care guidance.\n"
        "1. Identify the plant variety and current visible health symptoms.\n"
        "2. Diagnose likely causes of the damage.\n"
        "3. Give 3 urgent treatment steps to do today.\n"
        "4. Give a short tomorrow checklist with progress indicators.\n\n"
        "Answer in plain text only. Do not include any self-introduction or preamble. "
        "Do not use markdown, asterisks, or headings. Use simple numbered lines and line breaks only."
    )
    
    # กำหนดพารามิเตอร์แบบชัดเจน (data=..., mime_type=...) ป้องกันการประมวลผลติดขัด
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes, 
                mime_type="image/jpeg"
            ),
            recovery_prompt
        ]
    )
    flower_info = response.text.strip()
    return {"flower_info": flower_info}