from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import os

app = FastAPI()

model = YOLO("retinaldetachmentmodel.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Perform prediction
        results = model.predict(source=temp_file_path, save=False, verbose=False)
        os.remove(temp_file_path)  # Clean up the temporary file

        # Extract prediction details
        predicted_boxes = results[0].boxes
        class_ids = predicted_boxes.cls.cpu().numpy()
        class_names = [results[0].names[int(cls_id)] for cls_id in class_ids]

        if class_names:
            return {"class_name": class_names[0]}
        else:
            return {"message": "No objects detected"}

    except Exception as e:
        return {"error": str(e)}
