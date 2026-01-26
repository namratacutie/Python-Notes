import cv2
import numpy as np
import math
import random
import time
import os
import sys

# --- Diagnostic Info ---
print(f"--- DIAGNOSTIC ---")
print(f"Python Version: {sys.version}")
print(f"Script Path: {os.path.abspath(__file__)}")

# --- Robust MediaPipe Imports ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import face_detection as mp_face_detection
    HAS_MEDIAPIPE = True
    print("MediaPipe: Successfully loaded from standard path.")
except (ImportError, AttributeError):
    try:
        # Fallback for some installations
        import mediapipe.solutions.hands as mp_hands
        import mediapipe.solutions.face_detection as mp_face_detection
        HAS_MEDIAPIPE = True
        print("MediaPipe: Successfully loaded from secondary path.")
    except Exception as e:
        HAS_MEDIAPIPE = False
        print(f"MediaPipe: Error loading - {e}")
        print("Switching to MOUSE fallback. Hand gestures will not work.")

# --- Configuration ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Colors (B, G, R) for OpenCV - High Fidelity Palette
COLOR_CYAN_GLOW = (255, 255, 150)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (240, 250, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (50, 50, 255)
COLOR_BLUE_DEEP = (255, 50, 0)
COLOR_YELLOW = (0, 255, 255)

class FaceMask:
    def __init__(self, mask_path):
        self.use_mp = HAS_MEDIAPIPE
        if self.use_mp:
            try:
                self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            except Exception as e:
                print(f"Face Detection init failed: {e}. Falling back to Haar.")
                self.use_mp = False
        
        if not self.use_mp:
            # Fallback to Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Load and process Iron Man Mask
        self.mask_img = cv2.imread(mask_path)
        if self.mask_img is not None:
            # Chroma keying to remove green background
            hsv = cv2.cvtColor(self.mask_img, cv2.COLOR_BGR2HSV)
            # Define range of green color in HSV (approx)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            
            # Mask of green
            mask = cv2.inRange(hsv, lower_green, upper_green)
            # Inverse mask (keeping the iron man part)
            self.mask_alpha = cv2.bitwise_not(mask)
            
            # Extract just the mask part with transparency
            self.mask_img = cv2.bitwise_and(self.mask_img, self.mask_img, mask=self.mask_alpha)
            print("Successfully loaded Iron Man Mask and removed green background.")
        else:
            print(f"Warning: Could not load mask from {mask_path}")

    def apply(self, img):
        if self.mask_img is None:
            return img

        if self.use_mp:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(img_rgb)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    
                    # Calculate bounding box
                    x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
                    bw, bh = int(bbox.width * iw), int(bbox.height * ih)
                    
                    # Iron Man mask typically needs a bit of padding around the face for better fit
                    padding_w = int(bw * 0.25)
                    padding_h = int(bh * 0.35)
                    
                    mx, my = x - padding_w, y - padding_h
                    mw, mh = bw + (2 * padding_w), bh + (2 * padding_h)
                    
                    # Ensure within bounds
                    mx, my = max(0, mx), max(0, my)
                    mw, mh = min(mw, iw - mx), min(mh, ih - my)
                    
                    if mw > 0 and mh > 0:
                        self.overlay_mask(img, mx, my, mw, mh)
        else:
            # Haar fallback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                self.overlay_mask(img, x, y, w, h)
        return img

    def overlay_mask(self, img, x, y, w, h):
        # Resize mask and its alpha
        resized_mask = cv2.resize(self.mask_img, (w, h), interpolation=cv2.INTER_AREA)
        resized_alpha = cv2.resize(self.mask_alpha, (w, h), interpolation=cv2.INTER_AREA)
        
        # Get ROI
        roi = img[y:y+h, x:x+w]
        
        # Black-out the area of the mask in ROI
        roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(resized_alpha))
        
        # Combine ROI and resized mask
        dst = cv2.add(roi_bg, resized_mask)
        
        # Put back in main image
        img[y:y+h, x:x+w] = dst

def draw_cyber_brackets(img, x, y, w, h, color, length):
    th = 2
    # Corner brackets
    # TL
    cv2.line(img, (x, y), (x + length, y), color, th)
    cv2.line(img, (x, y), (x, y + length), color, th)
    # TR
    cv2.line(img, (x + w, y), (x + w - length, y), color, th)
    cv2.line(img, (x + w, y), (x + w, y + length), color, th)
    # BL
    cv2.line(img, (x, y + h), (x + length, y + h), color, th)
    cv2.line(img, (x, y + h), (x, y + h - length), color, th)
    # BR
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, th)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, th)
    
    # Scanning line
    scan_y = int(y + (time.time() * 200) % h)
    cv2.line(img, (x, scan_y), (x + w, scan_y), color, 1)

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.7, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.use_mouse = not HAS_MEDIAPIPE
        self.mouse_pos = (WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        self.mouse_click = False
        
        if self.use_mouse:
            cv2.namedWindow("Touch Designer Hand Tracking")
            cv2.setMouseCallback("Touch Designer Hand Tracking", self.mouse_callback)
            return

        try:
            self.hands = mp_hands.Hands(
                static_image_mode=self.mode,
                max_num_hands=self.max_hands,
                min_detection_confidence=detection_con,
                min_tracking_confidence=track_con
            )
        except Exception as e:
            print(f"MediaPipe failed to init: {e}. Switching to MOUSE mode.")
            self.use_mouse = True
            cv2.namedWindow("Touch Designer Hand Tracking")
            cv2.setMouseCallback("Touch Designer Hand Tracking", self.mouse_callback)

        self.tip_ids = [4, 8, 12, 16, 20]

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_click = False

    def find_hands(self, img):
        if self.use_mouse:
            cx, cy = self.mouse_pos
            landmarks = []
            for i in range(21):
                landmarks.append([i, cx + random.randint(-1,1), cy + random.randint(-1,1)])
            return [{"landmarks": landmarks, "center": (cx, cy), "size": 100}]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_landmarks = []
        
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                h, w, c = img.shape
                landmarks = []
                x_list = []
                y_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([id, cx, cy])
                    x_list.append(cx)
                    y_list.append(cy)
                
                all_landmarks.append({
                    "landmarks": landmarks,
                    "center": (int(np.mean(x_list)), int(np.mean(y_list))),
                    "size": math.hypot(landmarks[0][1] - landmarks[12][1], landmarks[0][2] - landmarks[12][2])
                })
        return all_landmarks

    def get_gesture(self, landmarks):
        if self.use_mouse:
            return ("CLOSED" if self.mouse_click else "OPEN"), 0

        fingers = []
        # Thumb
        if math.hypot(landmarks[4][1] - landmarks[9][1], landmarks[4][2] - landmarks[9][2]) > \
           math.hypot(landmarks[3][1] - landmarks[9][1], landmarks[3][2] - landmarks[9][2]):
             fingers.append(1)
        else: fingers.append(0)

        for id in range(1, 4): # Index, Middle, Ring
            if landmarks[self.tip_ids[id]][2] < landmarks[self.tip_ids[id] - 2][2]: fingers.append(1)
            else: fingers.append(0)
            
        total = fingers.count(1)
        if total <= 1: return "CLOSED", 0
        return "OPEN", 0

class GeometricFlower:
    def __init__(self):
        self.petal_count = 8
        self.rotation = 0
        self.bloom = 0

    def update(self, scale):
        self.rotation += 2.0 * scale
        self.bloom = math.sin(time.time() * 2) * 20 * scale

    def draw(self, img, cx, cy, color, scale):
        overlay = img.copy()
        max_r = 100 * scale
        
        # Draw 8 petals using polar parametric equations
        for i in range(self.petal_count):
            angle_p = math.radians(i * (360 / self.petal_count) + self.rotation)
            
            # Points for a "tear drop" or leaf petal
            points = []
            for deg in range(-45, 46, 5):
                rad = math.radians(deg)
                # Parametric petal shape
                r = (math.cos(rad) * 40 + self.bloom) * scale
                px = int(cx + math.cos(angle_p + rad * 0.5) * r * 2)
                py = int(cy + math.sin(angle_p + rad * 0.5) * r * 2)
                points.append((px, py))
            
            if len(points) > 1:
                pts = np.array(points, np.int32)
                cv2.polylines(overlay, [pts], False, color, 1)
                
                # Draw a glowing center line for each petal
                end_x = int(cx + math.cos(angle_p) * 60 * scale)
                end_y = int(cy + math.sin(angle_p) * 60 * scale)
                cv2.line(overlay, (cx, cy), (end_x, end_y), COLOR_WHITE, 1)

        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

class ComplexHUD:
    def __init__(self):
        self.angle_offset = 0
        self.arcs = []
        for i in range(15):
            self.arcs.append({
                "radius": random.randint(60, 220),
                "len": random.randint(10, 150),
                "speed": random.uniform(0.2, 1.5) * random.choice([1, -1]),
                "angle": random.randint(0, 360),
                "width": random.randint(1, 4),
                "layer": random.randint(1, 3)
            })
        self.data_nums = [f"{random.randint(0,9)}{random.randint(0,9)}" for _ in range(16)]
        self.flower = GeometricFlower()

    def update(self, scale):
        self.angle_offset += 0.8 * scale
        for arc in self.arcs:
            arc["angle"] += arc["speed"] * (scale + 0.5)
        self.flower.update(scale)

    def draw_glow(self, img, cx, cy, color, scale):
        overlay = img.copy()
        
        # New Flower Design
        self.flower.draw(img, cx, cy, color, scale)
        
        # Complex Geometry
        for arc in self.arcs:
            r = int(arc["radius"] * scale)
            start = int(arc["angle"])
            end = start + int(arc["len"] * (scale if arc["layer"] == 1 else 1.0))
            
            # Draw glow (thicker, lower alpha)
            cv2.ellipse(overlay, (cx, cy), (r, r), 0, start, end, color, arc["width"] + 2)
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, end, COLOR_WHITE, arc["width"])

        # Tick marks
        for i in range(0, 360, 5):
            rad = math.radians(i + self.angle_offset * 0.1)
            r1 = int(240 * scale)
            r2 = r1 + (8 if i % 30 == 0 else 4)
            x1, y1 = int(cx + math.cos(rad) * r1), int(cy + math.sin(rad) * r1)
            x2, y2 = int(cx + math.cos(rad) * r2), int(cy + math.sin(rad) * r2)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)

        # Dynamic Text
        for i, val in enumerate(self.data_nums):
            angle = i * (360/len(self.data_nums)) + self.angle_offset
            rad = math.radians(angle)
            r_txt = int(140 * scale)
            tx, ty = int(cx + math.cos(rad) * r_txt), int(cy + math.sin(rad) * r_txt)
            if random.random() < 0.05: self.data_nums[i] = f"{random.randint(0,9)}{random.randint(0,9)}"
            cv2.putText(img, val, (tx, ty), cv2.FONT_HERSHEY_PLAIN, 0.7, COLOR_WHITE, 1)

        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_WIDTH); cap.set(4, WINDOW_HEIGHT)
    
    detector = HandDetector(max_hands=1)
    mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "..", "..", "..", "..", "..", ".gemini", "antigravity", "brain", "6701b979-6ac2-493e-938d-51a6e0e978c0", "iron_man_mask_1769016879877.png")
    # Absolute path check for more reliability
    if not os.path.exists(mask_path):
        mask_path = "C:/Users/Lawarna/.gemini/antigravity/brain/6701b979-6ac2-493e-938d-51a6e0e978c0/iron_man_mask_1769016879877.png"
    
    face_sys = FaceMask(mask_path)
    hud = ComplexHUD()
    
    current_color = COLOR_CYAN
    scale_mult = 1.0
    
    print(f"--- RUNNING ---")
    print(f"Mode: {'HAND TRACKING' if not detector.use_mouse else 'MOUSE FALLBACK'}")
    print(f"MediaPipe Status: {'READY' if HAS_MEDIAPIPE else 'NOT FOUND'}")
    print("Press 'q' in the window to quit.")
    
    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Iron Man Mask Overlay
        img = face_sys.apply(img)
        
        hands = detector.find_hands(img)
        cx, cy = WINDOW_WIDTH//2, WINDOW_HEIGHT//2
        target_color = COLOR_CYAN
        target_scale = 1.0
        
        if hands:
            hand = hands[0]
            cx, cy = hand["center"]
            gesture, _ = detector.get_gesture(hand["landmarks"])
            
            if gesture == "CLOSED":
                target_color = COLOR_ORANGE
                target_scale = 0.6
                cv2.circle(img, (cx, cy), 180, COLOR_RED, 2) # Warning pulse
            else:
                target_color = COLOR_CYAN
                target_scale = 1.3
                
        # Interp
        scale_mult = 0.85 * scale_mult + 0.15 * target_scale
        c_new = [int(0.8 * c + 0.2 * t) for c, t in zip(current_color, target_color)]
        current_color = tuple(c_new)
        
        # Draw HUD
        hud.update(scale_mult)
        hud.draw_glow(img, cx, cy, current_color, scale_mult)
        
        # Hand specific details
        if hands and not detector.use_mouse:
            for pt in hand["landmarks"]:
                cv2.circle(img, (pt[1], pt[2]), 3, COLOR_WHITE, -1)
                cv2.line(img, (cx, cy), (pt[1], pt[2]), current_color, 1)

        # Status HUD
        mode_text = "HAND MODE" if not detector.use_mouse else "MOUSE MODE"
        cv2.putText(img, f"SYSTEM: {mode_text}", (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR_CYAN, 2)
        cv2.putText(img, "GESTURE: ACTIVE", (20, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, COLOR_WHITE, 1)
        
        cv2.imshow("Omniveral Tracking Studio", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
