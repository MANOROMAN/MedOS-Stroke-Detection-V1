import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import base64
import traceback
import albumentations as A
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# =============================================================================
# [CONFIG] SYSTEM CONFIGURATION
# NOTE(DevOps): Paths are relative for portability in containerized environments.
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Model Paths - Ensure these artifacts exist during deployment pipeline
PATH_DEEPLAB = os.path.join(MODEL_DIR, "best_deeplabv3_final.pth")
PATH_UNET = os.path.join(MODEL_DIR, "transfer_learning.pth")
PATH_YOLO = os.path.join(MODEL_DIR, "best_yolo11.pt")

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [CONFIG] SERVING FRONTEND (WEB)
# We configure Flask to serve the 'web' directory as static files at the '/web' URL path.
# This allows the user to access the interface at http://localhost:5000/web/stroke.html
app = Flask(__name__, static_folder='web', static_url_path='/web')
CORS(app) 

@app.route('/')
def index():
    """Redirect root to the main interface."""
    return "<script>window.location.href='/web/stroke.html';</script>"

# =============================================================================
# [CORE] CUSTOM ARCHITECTURE DEFINITIONS
# NOTE(ML-Team): Custom Attention U-Net implementation. 
# DO NOT MODIFY layer depth without updating pre-trained weights.
# =============================================================================

class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: [Conv2d -> BN -> ReLU] x 2
    """
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AttentionBlock(nn.Module):
    """
    Attention Gate for U-Net.
    Filters features propagated through skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    """
    Attention U-Net Architecture.
    TODO(Optimization): Consider pruning deeper layers for edge deployment.
    """
    def __init__(self, in_c=3, out_c=1, base=64, dropout=0.1):
        super().__init__()
        self.d1 = nn.Sequential(ConvBlock(in_c, base, dropout)); self.p1 = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(ConvBlock(base, base*2, dropout)); self.p2 = nn.MaxPool2d(2)
        self.d3 = nn.Sequential(ConvBlock(base*2, base*4, dropout)); self.p3 = nn.MaxPool2d(2)
        self.d4 = nn.Sequential(ConvBlock(base*4, base*8, dropout)); self.p4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*8, base*16, dropout)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionBlock(F_g=base*16, F_l=base*8, F_int=base*4)
        self.c1 = ConvBlock(base*8 + base*16, base*8, dropout)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionBlock(F_g=base*8, F_l=base*4, F_int=base*2)
        self.c2 = ConvBlock(base*4 + base*8, base*4, dropout)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionBlock(F_g=base*4, F_l=base*2, F_int=base)
        self.c3 = ConvBlock(base*2 + base*4, base*2, dropout)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att4 = AttentionBlock(F_g=base*2, F_l=base, F_int=base//2)
        self.c4 = ConvBlock(base + base*2, base, dropout)
        self.final = nn.Conv2d(base, out_c, 1)

    def forward(self, x):
        x1 = self.d1(x); p1 = self.p1(x1)
        x2 = self.d2(p1); p2 = self.p2(x2)
        x3 = self.d3(p2); p3 = self.p3(x3)
        x4 = self.d4(p3); p4 = self.p4(x4)
        b = self.bottleneck(p4)
        u1 = self.up1(b); x4 = self.att1(g=u1, x=x4); u1 = torch.cat([x4, u1], 1); u1 = self.c1(u1)
        u2 = self.up2(u1); x3 = self.att2(g=u2, x=x3); u2 = torch.cat([x3, u2], 1); u2 = self.c2(u2)
        u3 = self.up3(u2); x2 = self.att3(g=u3, x=x2); u3 = torch.cat([x2, u3], 1); u3 = self.c3(u3)
        u4 = self.up4(u3); x1 = self.att4(g=u4, x=x1); u4 = torch.cat([x1, u4], 1); u4 = self.c4(u4)
        return self.final(u4)

# =============================================================================
# [INIT] MODEL LOADER FACTORY
# =============================================================================
models = {}

def load_deeplab():
    if not os.path.exists(PATH_DEEPLAB): return None
    try:
        m = smp.DeepLabV3Plus(encoder_name="efficientnet-b5", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
        m.load_state_dict(torch.load(PATH_DEEPLAB, map_location=DEVICE, weights_only=False))
        m.eval()
        return m
    except Exception as e:
        print(f"[ERROR] DeepLab Initialization Failed: {e}")
        return None

def load_unet():
    if not os.path.exists(PATH_UNET): return None
    try:
        # ATTEMPT: High capacity model first
        m = AttentionUNet(in_c=3, out_c=1, base=128).to(DEVICE)
        try:
            m.load_state_dict(torch.load(PATH_UNET, map_location=DEVICE, weights_only=False))
        except:
            # FALLBACK: Low latency model
            print("[WARN] Base-128 weights mismatch. Falling back to Base-64 architecture.")
            m = AttentionUNet(in_c=3, out_c=1, base=64).to(DEVICE)
            m.load_state_dict(torch.load(PATH_UNET, map_location=DEVICE, weights_only=False))
        m.eval()
        return m
    except Exception as e:
        print(f"[ERROR] U-Net Initialization Failed: {e}")
        return None

def load_yolo():
    if not os.path.exists(PATH_YOLO): return None
    try:
        m = YOLO(PATH_YOLO)
        return m
    except Exception as e:
        print(f"[ERROR] YOLO Initialization Failed: {e}")
        return None

print(f"[SYSTEM] Initializing Inference Engines on {DEVICE.upper()}...")
models['deeplab'] = load_deeplab()
models['unet'] = load_unet()
models['yolo'] = load_yolo()
# NOTE(Strategy): Default to DeepLab for accuracy, fallback to U-Net
models['default'] = models['deeplab'] if models['deeplab'] else (models['unet'] if models['unet'] else None)
print("[SYSTEM] Initialization Complete.")

# --- HELPERS ---
val_transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])

def get_prob_map(model, img_tensor, model_type, original_shape):
    """
    Standardizes output from PyTorch models to Probability Map (0.0 - 1.0).
    """
    h, w = original_shape
    if model is None: return np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        if model_type == 'yolo':
             return np.zeros((h, w), dtype=np.float32) # YOLO handled separately
        else:
            out = model(img_tensor)
            prob = torch.sigmoid(out).cpu().numpy()[0, 0]
            prob = cv2.resize(prob, (w, h))
            return prob

def run_yolo_inference(model, img_bgr, original_shape):
    """
    Wrapper for Ultralytics YOLO inference.
    FIXME: Mask aggregation for overlapping instances needs improvement.
    """
    h, w = original_shape
    if model is None: return np.zeros((h, w), dtype=np.float32)
    try:
        results = model.predict(img_bgr, verbose=False, imgsz=IMG_SIZE)
        if len(results) > 0 and results[0].masks is not None:
             # Aggregating all masks
             any_mask = results[0].masks.data.sum(dim=0).clamp(0,1).cpu().numpy()
             return cv2.resize(any_mask, (w, h))
        return np.zeros((h, w), dtype=np.float32)
    except Exception as e:
        print(f"[WARN] YOLO Inference Exception: {e}")
        return np.zeros((h, w), dtype=np.float32)


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def process_request(image_bytes, selected_model, viz_mode='fill', color_hex='#ff0000', glow=False):
    """
    Core pipeline: Preprocessing -> Inference -> Postprocessing -> Visualization.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None: raise ValueError("Image Decoding Failed - Corrupt Buffer?")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Preprocessing
    aug = val_transform(image=img_rgb)
    img_tensor = aug['image'].unsqueeze(0).to(DEVICE)
    
    final_prob = np.zeros((h, w), dtype=np.float32)
    
    # [INFERENCE STRATEGY]
    if selected_model == 'ensemble':
        # NOTE(Research): Averaging outputs stabilizes variance between architectures.
        p1 = get_prob_map(models['deeplab'], img_tensor, 'torch', (h,w))
        p2 = get_prob_map(models['unet'], img_tensor, 'torch', (h,w))
        p3 = run_yolo_inference(models['yolo'], img_bgr, (h,w))
        final_prob = (p1 + p2 + p3) / 3.0
    
    elif selected_model == 'yolo':
        final_prob = run_yolo_inference(models['yolo'], img_bgr, (h,w))
        
    elif selected_model == 'unet':
        final_prob = get_prob_map(models['unet'], img_tensor, 'torch', (h,w))
        
    else: # Default: DeepLab efficiency
        final_prob = get_prob_map(models['deeplab'], img_tensor, 'torch', (h,w))

    # [POST-PROCESSING] Thresholding & Noise Reduction
    mask_bin = (final_prob > 0.5).astype(np.uint8)
    
    # Cleaning small artifacts
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 50: mask_bin[labels==i] = 0

    # [VISUALIZATION]
    viz_img = img_rgb.copy()
    
    if viz_mode != 'hide' and np.any(mask_bin):
        try:
            target_color = np.array(hex_to_bgr(color_hex)[::-1], dtype=np.uint8) 
        except:
            target_color = np.array([255, 0, 0], dtype=np.uint8) 

        # Experimental: Neon Glow Effect
        if glow:
            mask_glow = np.zeros_like(viz_img)
            if viz_mode == 'outline':
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask_glow, contours, -1, target_color.tolist(), 5)
            else:
                mask_glow[mask_bin == 1] = target_color
            blurred_glow = cv2.GaussianBlur(mask_glow, (21, 21), 0)
            viz_img = cv2.addWeighted(viz_img, 1.0, blurred_glow, 0.8, 0)

        if viz_mode == 'fill':
            alpha = 0.4
            mask_indices = mask_bin == 1
            roi = viz_img[mask_indices].astype(np.float32)
            blended = roi * (1 - alpha) + target_color.astype(np.float32) * alpha
            viz_img[mask_indices] = blended.astype(np.uint8)
            
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_color = (255, 255, 255) if glow else (255, 255, 0)
            cv2.drawContours(viz_img, contours, -1, contour_color, 2)
            
        elif viz_mode == 'outline':
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color_c = target_color.tolist()
            th = 3 if glow else 2
            cv2.drawContours(viz_img, contours, -1, color_c, th)

    # Encode for Client
    vis_bgr = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', vis_bgr)
    str_vis = base64.b64encode(buffer).decode('utf-8')
    
    # Calculate Risk Metrix
    area_px = int(np.sum(mask_bin))
    if area_px > 0:
        # Confidence Metric: Mean probability relative to segmentation area
        mean_conf = final_prob[mask_bin == 1].mean()
        risk = float(mean_conf * 100)
    else:
        risk = 10.0 # Baseline background noise risk
    
    return str_vis, area_px, risk

# =============================================================================
# [API] ENDPOINTS
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Inference Endpoint. 
    Accepts: Multipart-encoded image ('file')
    Returns: JSON with base64 visualization and metrics.
    """
    try:
        if 'file' not in request.files: return jsonify({'error': 'Payload Missing File'}), 400
        file = request.files['file']
        model_req = request.form.get('model', 'default')
        viz_mode = request.form.get('viz_mode', 'fill')
        color_hex = request.form.get('color', '#ff0000')
        glow_req = request.form.get('glow', 'false').lower() == 'true'
        
        viz_b64, area, risk = process_request(file.read(), model_req, viz_mode, color_hex, glow_req)
        
        return jsonify({
            'visualization': f"data:image/png;base64,{viz_b64}",
            'has_stroke': area > 0,
            'area': area,
            'risk_score': risk,
            'model_used': model_req
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/save_report', methods=['POST'])
def save_report():
    """
    Report Archival Endpoint.
    Saves the approved analysis snapshot to the local filesystem.
    TODO(Database): Move from file-based storage to Blob Storage (S3/Azure).
    """
    try:
        data = request.json
        img_b64 = data.get('image', '').split(',')[-1]
        risk = data.get('risk', 0)
        area = data.get('area', 0)
        
        if not img_b64: return jsonify({'error': 'No content provided'}), 400

        if not os.path.exists(REPORT_DIR): os.makedirs(REPORT_DIR)
        
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"Report_{timestamp}_Risk{risk}_{area}px.png"
        filepath = os.path.join(REPORT_DIR, filename)
        
        with open(filepath, "wb") as fh:
            fh.write(base64.b64decode(img_b64))
            
        return jsonify({'status': 'success', 'path': filepath})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"[BOOT] MedOS Server Online.")
    print(f"[BOOT] Access UI at: http://localhost:5000/web/stroke.html")
    app.run(host='0.0.0.0', port=5000)
