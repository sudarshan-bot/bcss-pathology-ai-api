
import base64
import datetime
import io
import os
import tempfile

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
    KeepTogether,
)

app = FastAPI(
    title="BCSS Pathology AI API",
    description="Research prototype for histopathology segmentation and PDF reporting",
    version="1.1.0",
)

CLASS_NAMES = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflammatory",
    3: "Necrosis",
    4: "Others",
}

# RGB colors for the segmentation overlay
COLORS = np.array(
    [
        [255, 0, 0],       # Tumour: red
        [0, 200, 0],       # Stroma: green
        [255, 165, 0],     # Inflammatory: orange
        [160, 32, 240],    # Necrosis: purple
        [0, 180, 255],     # Others: blue
    ],
    dtype=np.uint8,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

segmentor = None
model_error = None


@app.on_event("startup")
def load_model():
    """Initialize the TIAToolbox pretrained BCSS model."""
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
        model_error = str(exc)
        print("Model initialization failed:", repr(exc))


@app.get("/")
def home():
    return {
        "message": "BCSS Pathology AI API",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
        "pdf_endpoint": "/predict-report",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": segmentor is not None,
        "model_error": model_error,
    }


def encode_png(array, mode):
    """Encode a NumPy image as a Base64 PNG string."""
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run_segmentation(image):
    """
    Run the pretrained segmentation model on a PIL RGB image.
    Returns the mask, colored overlay, and class pixel counts.
    """
    if segmentor is None:
        raise RuntimeError(
            f"Model is unavailable: {model_error or 'not initialized'}"
        )

    original_width, original_height = image.size

    # This pretrained model expects 1024 x 1024 input.
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

        pred = np.asarray(output["predictions"][0])

    pred = np.squeeze(pred)

    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

    # Resize categorical labels using nearest-neighbor interpolation.
    mask_image = Image.fromarray(pred.astype(np.uint8), mode="L")
    mask_image = mask_image.resize(
        (original_width, original_height),
        Image.Resampling.NEAREST,
    )
    mask = np.asarray(mask_image)

    original = np.asarray(image.convert("RGB")).copy()

    # Create a colored mask and blend it with the original image.
    valid = (mask >= 0) & (mask < len(COLORS))
    colored_mask = np.zeros_like(original)
    colored_mask[valid] = COLORS[mask[valid]]

    overlay = original.copy()
    overlay[valid] = (
        0.55 * original[valid] + 0.45 * colored_mask[valid]
    ).astype(np.uint8)

    class_counts = {
        CLASS_NAMES[class_id]: int(np.count_nonzero(mask == class_id))
        for class_id in CLASS_NAMES
    }

    return mask, overlay, class_counts


def create_pdf_report(
    original_bytes,
    original_image,
    mask,
    overlay,
    class_counts,
    filename,
):
    """Build the PDF report in memory and return its bytes."""
    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="BCSS Pathology AI Research Report",
        author="BCSS Pathology AI API",
    )

    styles = getSampleStyleSheet()
    story = []

    width, height = original_image.size
    generated_at = datetime.datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    story.append(Paragraph("BCSS Pathology AI Report", styles["Title"]))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "Research prototype — not a clinical diagnosis",
            styles["Heading2"],
        )
    )
    story.append(Paragraph(f"Filename: {filename}", styles["Normal"]))
    story.append(
        Paragraph(
            f"Original image dimensions: {width} × {height} pixels",
            styles["Normal"],
        )
    )
    story.append(Paragraph(f"Report generated: {generated_at}", styles["Normal"]))
    story.append(Spacer(1, 5 * mm))

    # Original uploaded image
    original_buffer = io.BytesIO(original_bytes)
    story.append(Paragraph("1. Original Pathology Image", styles["Heading2"]))
    story.append(
        PDFImage(
            original_buffer,
            width=170 * mm,
            height=115 * mm,
            kind="proportional",
        )
    )
    story.append(Spacer(1, 5 * mm))

    # Colored segmentation overlay
    overlay_buffer = io.BytesIO()
    Image.fromarray(overlay, mode="RGB").save(overlay_buffer, format="PNG")
    overlay_buffer.seek(0)

    story.append(Paragraph("2. Segmentation Overlay", styles["Heading2"]))
    story.append(
        PDFImage(
            overlay_buffer,
            width=170 * mm,
            height=115 * mm,
            kind="proportional",
        )
    )
    story.append(Spacer(1, 4 * mm))

    # Predicted categorical mask
    mask_buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(
        mask_buffer, format="PNG"
    )
    mask_buffer.seek(0)

    story.append(Paragraph("3. Predicted Segmentation Mask", styles["Heading2"]))
    story.append(
        Paragraph(
            "The mask stores the predicted class ID for each pixel. "
            "The color legend below describes the overlay colors.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        PDFImage(
            mask_buffer,
            width=170 * mm,
            height=115 * mm,
            kind="proportional",
        )
    )
    story.append(Spacer(1, 4 * mm))

    # Class legend
    story.append(Paragraph("4. Class Legend", styles["Heading2"]))

    legend_data = [["Class ID", "Class", "Overlay color"]]
    color_names = ["Red", "Green", "Orange", "Purple", "Blue"]

    for class_id, class_name in CLASS_NAMES.items():
        legend_data.append([str(class_id), class_name, color_names[class_id]])

    legend_table = Table(
        legend_data,
        colWidths=[25 * mm, 70 * mm, 55 * mm],
        repeatRows=1,
    )
    legend_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F0F4F8")],
                ),
            ]
        )
    )
    story.append(legend_table)
    story.append(Spacer(1, 5 * mm))

    # Pixel counts and percentages
    story.append(Paragraph("5. Predicted Class Statistics", styles["Heading2"]))

    total_pixels = width * height
    table_data = [["Class", "Pixel count", "Image percentage"]]

    for class_name, count in class_counts.items():
        percentage = (
            (count / total_pixels) * 100 if total_pixels else 0
        )
        table_data.append(
            [class_name, f"{count:,}", f"{percentage:.2f}%"]
        )

    stats_table = Table(
        table_data,
        colWidths=[60 * mm, 45 * mm, 50 * mm],
        repeatRows=1,
    )
    stats_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F0F4F8")],
                ),
            ]
        )
    )
    story.append(stats_table)
    story.append(Spacer(1, 5 * mm))

    # Measurement limitations
    story.append(Paragraph("6. Measurement Notes", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Image dimensions: {width} × {height} pixels. "
            "Pixel counts describe model-predicted labels. This report does "
            "not calculate verified individual tumor-region boundaries, "
            "lengths, widths, or perimeters. Such measurements require a "
            "separate region-analysis procedure. Physical dimensions in "
            "micrometers or millimeters cannot be calculated without a "
            "reliable image scale or microscope calibration.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("7. Research Disclaimer", styles["Heading2"]))
    story.append(
        Paragraph(
            "This report presents experimental outputs from a pretrained "
            "semantic segmentation model. Predicted classes are not confirmed "
            "diagnoses. Model performance, class mapping, and suitability "
            "for clinical use have not been established by this API. Do not "
            "use this report as a standalone diagnostic tool; interpretation "
            "requires a qualified pathologist.",
            styles["BodyText"],
        )
    )

    document.build(story)
    result = pdf_buffer.getvalue()
    pdf_buffer.close()
    return result


async def read_uploaded_image(file: UploadFile):
    """Read and validate an uploaded image once."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Upload a valid image file.",
        )

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 20 MB or smaller.",
        )

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.load()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )
    finally:
        await file.close()

    return contents, image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Return segmentation results as JSON with Base64 PNG images."""
    if segmentor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {model_error or 'not initialized'}",
        )

    _, image = await read_uploaded_image(file)

    try:
        mask, overlay, class_counts = run_segmentation(image)

        return {
            "filename": file.filename,
            "image_width": image.width,
            "image_height": image.height,
            "class_names": CLASS_NAMES,
            "pixel_counts": class_counts,
            "mask_png_base64": encode_png(mask, "L"),
            "overlay_png_base64": encode_png(overlay, "RGB"),
            "notice": (
                "Research prototype only. Results are not clinically "
                "validated and must not be used as a standalone diagnosis."
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


@app.post("/predict-report")
async def predict_report(file: UploadFile = File(...)):
    """Run segmentation and return a downloadable PDF report."""
    if segmentor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {model_error or 'not initialized'}",
        )

    original_bytes, image = await read_uploaded_image(file)

    try:
        mask, overlay, class_counts = run_segmentation(image)

        pdf_bytes = create_pdf_report(
            original_bytes=original_bytes,
            original_image=image,
            mask=mask,
            overlay=overlay,
            class_counts=class_counts,
            filename=file.filename or "uploaded_image",
        )

        safe_filename = "bcss_pathology_report.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_filename}"'
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        print("PDF report generation failed:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate PDF report: {str(exc)}",
        )
