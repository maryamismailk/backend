
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pathlib import Path
import uuid
import requests

from database import SessionLocal, engine, Base
import models


app = FastAPI(title="AgriBot Backend")


# Create database tables
Base.metadata.create_all(bind=engine)


# Create folder to store images
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "AgriBot Backend is running!"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "OK"
    }


# Upload image and analyze it using AI
@app.post("/upload-image")
async def upload_image(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):

    # Check that the uploaded file is an image
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed"
        )

    # Read image data
    image_data = await image.read()

    # Check image size (maximum 10 MB)
    if len(image_data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large"
        )

    # Generate a unique filename
    extension = ".jpg" if image.content_type == "image/jpeg" else ".png"
    filename = f"{uuid.uuid4()}{extension}"

    # Save image
    image_path = UPLOAD_FOLDER / filename
    image_path.write_bytes(image_data)

    # Connect to database
    db = SessionLocal()

    try:

        # Create a new inspection record
        new_inspection = models.Inspection(
            image_path=f"uploads/{filename}",
            latitude=latitude,
            longitude=longitude,
            status="PROCESSING"
        )

        # Save inspection in MySQL
        db.add(new_inspection)
        db.commit()
        db.refresh(new_inspection)

        # Send saved image to JavaScript AI service
        try:

            with open(image_path, "rb") as image_file:

                ai_response = requests.post(
                    "http://localhost:3000/predict",
                    files={
                        "image": (
                            filename,
                            image_file,
                            image.content_type
                        )
                    },
                    timeout=30
                )

            # Check AI response
            ai_response.raise_for_status()
            ai_result = ai_response.json()

            # Get prediction results
            classification = ai_result["classification"]
            confidence = float(ai_result["confidence"])

            # Determine inspection decision
            if classification == "No_Leaf_or_Unclear":
                decision = "Unclear image - retake required"

            elif confidence < 70:
                decision = "Low confidence - review required"

            else:
                decision = classification

            # Update existing inspection record
            new_inspection.classification = classification
            new_inspection.confidence = confidence
            new_inspection.decision = decision
            new_inspection.status = "COMPLETED"

            db.commit()
            db.refresh(new_inspection)

        except Exception as e:

            # AI failed, but keep the uploaded image and record
            print("AI ERROR:", repr(e))

            new_inspection.status = "FAILED"
            db.commit()

        # Return inspection information
        return {
            "status": "success",
            "message": "Image uploaded and inspection saved!",
            "inspection_id": new_inspection.id,
            "image_path": f"uploads/{filename}",
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "classification": new_inspection.classification,
            "confidence": new_inspection.confidence,
            "decision": new_inspection.decision,
            "processing_status": new_inspection.status
        }

    except Exception as e:

        # Handle database errors
        db.rollback()

        print("DATABASE ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail="Failed to save inspection"
        )

    finally:
        db.close()
