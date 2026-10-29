
import base64
import io
import os
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
)

app = FastAPI(
    title="BCSS Pathology AI API",
    description="Research prototype for histopathology segmentation",
    version="1.3.0",
)

CLASS_NAMES = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflammatory",
    3: "Necrosis",
    4: "Others",
}

COLORS = np.array(
    [
        [255, 0, 0],
        [0, 200, 0],
        [255, 165, 0],
        [160, 32, 240],
        [0, 180, 255],
    ],
    dtype=np.uint8,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

segmentor = None
model_error = None


@app.on_event("startup")
def load_model():
    """Load TIAToolbox's pretrained BCSS segmentation model."""
    global segmentor, model_error

    try:
        segmentor = SemanticSegmentor(
            model="fcn_resnet50_unet-bcss",
            num_workers=0,
            batch_size=1,
        )
        model_error = None
        print("BCSS model initialized successfully.")

    except Exception as exc:
        segmentor = None
        model_error = f"{type(exc).__name__}: {exc}"
        print("Model initialization failed:", model_error)


@app.get("/")
def home():
    return {
        "message": "BCSS Pathology AI API",
        "health": "/health",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
        "pdf_endpoint": "/predict-report",
        "notice": "Research prototype, not a clinical diagnosis.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": segmentor is not None,
        "model_error": model_error,
    }


def encode_png(array, mode):
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run_segmentation(image):
    if segmentor is None:
        raise RuntimeError(
            f"Model unavailable: {model_error or 'not initialized'}"
        )

    original_width, original_height = image.size

    # The pretrained model expects 1024 x 1024 input.
    model_image = image.resize(
        (1024, 1024),
        Image.Resampling.BILINEAR,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.png")
        model_image.save(input_path)

        output = segmentor.run(
            images=[input_path],
            patch_mode=True,
            output_type="dict",
            num_workers=0,
            batch_size=1,
        )

        prediction = np.asarray(output["predictions"][0])

    prediction = np.squeeze(prediction)

    if prediction.ndim != 2:
        raise RuntimeError(
            f"Unexpected prediction shape: {prediction.shape}"
        )

    mask_image = Image.fromarray(prediction.astype(np.uint8), mode="L")
    mask_image = mask_image.resize(
        (original_width, original_height),
        Image.Resampling.NEAREST,
    )
    mask = np.asarray(mask_image)

    original = np.asarray(image.convert("RGB")).copy()

    valid = (mask >= 0) & (mask < len(COLORS))
    colored_mask = np.zeros_like(original)
    colored_mask[valid] = COLORS[mask[valid]]

    overlay = original.copy()
    overlay[valid] = (
        0.55 * original[valid] + 0.45 * colored_mask[valid]
    ).astype(np.uint8)

    pixel_counts = {
        name: int(np.count_nonzero(mask == class_id))
        for class_id, name in CLASS_NAMES.items()
    }

    return mask, overlay, pixel_counts


async def read_uploaded_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 20 MB or smaller.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.load()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )
    finally:
        await file.close()

    return contents, image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if segmentor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_error or 'not initialized'}",
        )

    _, image = await read_uploaded_image(file)

    try:
        mask, overlay, pixel_counts = run_segmentation(image)

        return {
            "filename": file.filename,
            "image_width": image.width,
            "image_height": image.height,
            "class_names": CLASS_NAMES,
            "pixel_counts": pixel_counts,
            "mask_png_base64": encode_png(mask, "L"),
            "overlay_png_base64": encode_png(overlay, "RGB"),
            "notice": (
                "Research output only. Predictions are not clinically "
                "validated and are not a standalone diagnosis."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("Prediction failed:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(exc)}",
        )


def create_pdf_report(original_bytes, image, mask, overlay, pixel_counts, filename):
    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph("BCSS Pathology AI Report", styles["Title"]),
        Spacer(1, 4 * mm),
        Paragraph(
            "Research prototype — not a clinical diagnosis",
            styles["Heading2"],
        ),
        Paragraph(f"Filename: {filename}", styles["Normal"]),
        Paragraph(
            f"Image size: {image.width} x {image.height} pixels",
            styles["Normal"],
        ),
        Paragraph(
            f"Generated: {datetime.now().astimezone().isoformat()}",
            styles["Normal"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("Original Image", styles["Heading2"]),
    ]

    original_buffer = io.BytesIO(original_bytes)
    story.append(
        PDFImage(
            original_buffer,
            width=170 * mm,
            height=110 * mm,
            kind="proportional",
        )
    )

    overlay_buffer = io.BytesIO()
    Image.fromarray(overlay).save(overlay_buffer, format="PNG")
    overlay_buffer.seek(0)

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Segmentation Overlay", styles["Heading2"]))
    story.append(
        PDFImage(
            overlay_buffer,
            width=170 * mm,
            height=110 * mm,
            kind="proportional",
        )
    )

    mask_buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(
        mask_buffer, format="PNG"
    )
    mask_buffer.seek(0)

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Predicted Mask", styles["Heading2"]))
    story.append(
        PDFImage(
            mask_buffer,
            width=170 * mm,
            height=110 * mm,
            kind="proportional",
        )
    )

    table_data = [["Class", "Pixel count"]]
    for name, count in pixel_counts.items():
        table_data.append([name, f"{count:,}"])

    table = Table(table_data, colWidths=[80 * mm, 60 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph("Predicted Class Statistics", styles["Heading2"]),
            table,
            Spacer(1, 5 * mm),
            Paragraph(
                "These are experimental segmentation results. They have "
                "not been clinically validated. Pixel counts are not "
                "physical measurements. A qualified pathologist must "
                "interpret the images.",
                styles["BodyText"],
            ),
        ]
    )

    document.build(story)
    result = pdf_buffer.getvalue()
    pdf_buffer.close()
    return result


@app.post("/predict-report")
async def predict_report(file: UploadFile = File(...)):
    if segmentor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_error or 'not initialized'}",
        )

    original_bytes, image = await read_uploaded_image(file)

    try:
        mask, overlay, pixel_counts = run_segmentation(image)

        pdf_bytes = create_pdf_report(
            original_bytes,
            image,
            mask,
            overlay,
            pixel_counts,
            file.filename or "uploaded_image",
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    'attachment; filename="bcss_pathology_report.pdf"'
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        print("PDF report failed:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate PDF: {str(exc)}",
        )
