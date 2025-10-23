import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from dataset_polygon import char2idx, idx2char
from model_cnn_transformer import OCRModel

# Model configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = len(char2idx)
OCR_MODEL_PATH = "best_ocr_model.pth"
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"
FONT_PATH = "Roboto-Regular.ttf"


def preprocess_ocr_image(pil_img):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(pil_img).unsqueeze(0)


def decode_sequence(indices):
    chars = []
    SOS_TOKEN = next((token for token in char2idx.keys() if "SOS" in token), None)
    for idx in indices:
        ch = idx2char.get(idx, "")
        if ch == "<EOS>":
            break
        if ch not in ("<PAD>", SOS_TOKEN):
            chars.append(ch)
    return "".join(chars)


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Load YOLO model
        yolo_model = YOLO(YOLO_MODEL_PATH)

        # Load OCR model
        ocr_model = OCRModel(vocab_size=VOCAB_SIZE).to(DEVICE)
        state_dict = torch.load(OCR_MODEL_PATH, map_location=DEVICE)
        ocr_model.load_state_dict(state_dict)
        ocr_model.eval()

        return yolo_model, ocr_model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


def draw_vietnamese_boxes_text(img_pil, bboxes, texts, font_size=22):
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, font_size)
    for (x1, y1, x2, y2), txt in zip(bboxes, texts):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, max(y1 - 22, 0)), txt, fill=(255, 0, 0), font=font)
    return img_pil


def yolo_ocr_pipeline(image_pil, yolo_model, ocr_model, conf_thresh=0.5):
    # Keep original image size for display
    orig_w, orig_h = image_pil.size

    # Detection
    results = yolo_model(image_pil)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    filtered = [(box, conf) for box, conf in zip(boxes, confs) if conf > conf_thresh]

    if not filtered:
        return [], []

    boxes = [box for box, _ in filtered]
    texts = []
    bboxes = []

    for box in boxes:
        try:
            x1, y1, x2, y2 = map(int, box)
            # Ensure valid box coordinates
            x1, x2 = max(0, x1), min(orig_w, x2)
            y1, y2 = max(0, y1), min(orig_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_pil.crop((x1, y1, x2, y2))
            image_tensor = preprocess_ocr_image(crop).to(DEVICE)

            with torch.no_grad():
                memory = ocr_model.encoder(image_tensor)
                SOS_TOKEN = next(
                    (token for token in char2idx.keys() if "SOS" in token), None
                )
                MAX_LEN = 36
                ys = torch.tensor([[char2idx[SOS_TOKEN]]], device=DEVICE)

                for _ in range(MAX_LEN):
                    out = ocr_model.decoder(
                        ys,
                        memory,
                        tgt_mask=ocr_model.generate_square_subsequent_mask(
                            ys.size(1)
                        ).to(DEVICE),
                    )
                    prob = out[:, -1, :]
                    _, next_word = torch.max(prob, dim=1)
                    ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

                    if next_word.item() == char2idx["<EOS>"]:
                        break

                pred_text = decode_sequence(ys.squeeze(0).tolist())
                if pred_text.strip():  # Only add non-empty text
                    texts.append(pred_text)
                    bboxes.append((x1, y1, x2, y2))
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")
            continue

    return bboxes, texts


# --- Streamlit App ---
def main():
    st.title("Vietnamese OCR Demo (Detection + Recognition)")
    st.write(
        "Upload a picture. The model will detect text regions and extract Vietnamese text."
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        try:
            # Read image
            img_pil = Image.open(uploaded_file).convert("RGB")
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Loading models..."):
                yolo_model, ocr_model = load_models()

            if yolo_model is None or ocr_model is None:
                st.error("Failed to load models. Please check your model paths.")
            else:
                with st.spinner("Detecting and Recognizing Text..."):
                    try:
                        bboxes, texts = yolo_ocr_pipeline(
                            img_pil, yolo_model, ocr_model, conf_thresh=0.5
                        )
                        if bboxes and texts:
                            img_draw = draw_vietnamese_boxes_text(
                                img_pil.copy(), bboxes, texts, font_size=22
                            )
                            st.image(
                                img_draw,
                                caption="Detected Text & OCR",
                                use_container_width=True,
                            )

                            st.markdown("### Kết quả nhận diện văn bản")
                            for i, (bbox, txt) in enumerate(zip(bboxes, texts), 1):
                                st.write(f"**{i}. [Box {bbox}]**: {txt}")
                        else:
                            st.warning("Không tìm thấy text trong ảnh.")
                    except Exception as e:
                        st.error(f"Lỗi khi nhận diện text: {str(e)}")
        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")


if __name__ == "__main__":
    main()
