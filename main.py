# main.py
from fastapi import FastAPI, UploadFile, File, Form, Request
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
@app.post("/find-nearest")
async def find_nearest_flower_garden(
    file: UploadFile = File(...),
    latitude: str = Form(None),
    longitude: str = Form(None)
):
    image_bytes = await file.read()

    location_note = ''
    if latitude and longitude:
        location_note = f"Use the user's approximate location: latitude {latitude}, longitude {longitude}."

    nearest_prompt = (
        f"Identify the flower in the provided photo and then suggest the nearest public flower garden, botanical garden, or park where this flower can likely be found near the user's location.\n"
        f"{location_note}\n"
        "Do not provide care instructions or diagnosis. Keep the answer concise and relevant to nearby flower spots."
    )

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            nearest_prompt
        ]
    )
    flower_info = response.text.strip()
    return {"flower_info": flower_info}
@app.post("/identify-type")
async def identify_flower_type(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    type_prompt = (
        "Analyze the photo and identify the flower species or common flower name only.\n"
        "Do not give care instructions, diagnosis, or additional commentary.\n"
        "Respond in plain text only with the flower type/name."
    )

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            type_prompt
        ]
    )
    flower_info = response.text.strip()
    return {"flower_info": flower_info}

@app.post("/identify-HSR")
async def identify_HSR_musume(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    HSR_prompt = (
        "Analyze the provided image and identify which honkai star rail character is shown.\n"
        "Answer with the character name only.\n"
        "Do not describe plants, flowers, or any other unrelated details.\n"
        "Respond in plain text only with the character name."
    )

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            HSR_prompt
        ]
    )
    HSR_info = response.text.strip()
    return {"HSR_info": HSR_info}