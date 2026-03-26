import math
import time
import random
from dataclasses import dataclass
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# ShinobiVision Pro
# Real-time Naruto-inspired gesture + FX demo in pure Python
# Tech: OpenCV + MediaPipe + NumPy
# ============================================================

# ------------------------------
# Config
# ------------------------------
CAMERA_INDEX = 0
WINDOW_NAME = "ShinobiVision Pro"
MAX_HANDS = 2
GESTURE_HISTORY = 8
HOLD_FRAMES = 5
GLOBAL_COOLDOWN = 0.7
FRAME_W = 1280
FRAME_H = 720
DRAW_LANDMARKS = True
ENABLE_RECORDING = False
OUTPUT_FILE = "shinobi_vision_demo.mp4"


# ------------------------------
# MediaPipe Setup
# ------------------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles


# ------------------------------
# Landmark indices
# ------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


# ------------------------------
# Data classes
# ------------------------------
@dataclass
class HandState:
    label: str
    handedness_score: float
    pts_px: np.ndarray          # (21, 2) pixel coordinates
    pts_norm: np.ndarray        # (21, 2) normalized to wrist/scale
    finger_bits: Tuple[int, int, int, int, int]
    pinch_index: float
    pinch_middle: float
    palm_size: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    born: float
    size: float
    color: Tuple[int, int, int]
    kind: str = "normal"

    @property
    def alive(self) -> bool:
        return (time.time() - self.born) < self.life

    @property
    def age_ratio(self) -> float:
        return min(1.0, (time.time() - self.born) / max(self.life, 1e-6))


# ------------------------------
# Utility math helpers
# ------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def draw_glow_circle(img: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], layers: int = 4):
    overlay = np.zeros_like(img)
    for i in range(layers, 0, -1):
        r = max(1, int(radius * (i / layers)))
        cv2.circle(overlay, center, r, color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.20, img, 1.0, 0, img)


def draw_glow_line(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2):
    overlay = np.zeros_like(img)
    for t in [thickness + 6, thickness + 3, thickness]:
        cv2.line(overlay, p1, p2, color, t, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, img, 1.0, 0, img)


def put_title(frame: np.ndarray, text: str, color=(50, 255, 255)):
    cv2.rectangle(frame, (18, 14), (420, 54), (0, 0, 0), -1)
    cv2.putText(frame, text, (26, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def put_subtitle(frame: np.ndarray, text: str, y: int, color=(220, 220, 220), scale: float = 0.55):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2, cv2.LINE_AA)


def smooth_point(history: Deque[Tuple[int, int]]) -> Tuple[int, int]:
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


# ------------------------------
# Feature extraction
# ------------------------------
def hand_to_arrays(hand_landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    return np.array(pts, dtype=np.int32)


def normalize_hand(pts_px: np.ndarray) -> Tuple[np.ndarray, float]:
    wrist = pts_px[WRIST].astype(np.float32)
    middle_mcp = pts_px[MIDDLE_MCP].astype(np.float32)
    palm_size = max(np.linalg.norm(middle_mcp - wrist), 1.0)
    norm = (pts_px.astype(np.float32) - wrist) / palm_size
    return norm, float(palm_size)


def infer_fingers(pts: np.ndarray, label: str) -> Tuple[int, int, int, int, int]:
    fingers = []

    if label == "Right":
        thumb_open = 1 if pts[THUMB_TIP][0] > pts[THUMB_IP][0] else 0
    else:
        thumb_open = 1 if pts[THUMB_TIP][0] < pts[THUMB_IP][0] else 0
    fingers.append(thumb_open)

    fingers.append(1 if pts[INDEX_TIP][1] < pts[INDEX_PIP][1] else 0)
    fingers.append(1 if pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1] else 0)
    fingers.append(1 if pts[RING_TIP][1] < pts[RING_PIP][1] else 0)
    fingers.append(1 if pts[PINKY_TIP][1] < pts[PINKY_PIP][1] else 0)
    return tuple(fingers)


def build_hand_state(hand_landmarks, handedness, frame_shape) -> HandState:
    h, w = frame_shape[:2]
    pts = hand_to_arrays(hand_landmarks, w, h)
    pts_norm, palm_size = normalize_hand(pts)
    label = handedness.classification[0].label
    score = float(handedness.classification[0].score)
    finger_bits = infer_fingers(pts, label)

    pinch_index = l2(pts_norm[THUMB_TIP], pts_norm[INDEX_TIP])
    pinch_middle = l2(pts_norm[THUMB_TIP], pts_norm[MIDDLE_TIP])

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    center = tuple(np.mean(pts, axis=0).astype(int))

    return HandState(
        label=label,
        handedness_score=score,
        pts_px=pts,
        pts_norm=pts_norm,
        finger_bits=finger_bits,
        pinch_index=pinch_index,
        pinch_middle=pinch_middle,
        palm_size=palm_size,
        center=center,
        bbox=(int(x1), int(y1), int(x2), int(y2)),
    )


def extract_eye_positions(face_landmarks, frame_shape) -> List[Tuple[int, int]]:
    h, w = frame_shape[:2]
    pts = face_landmarks.landmark
    result = []
    for idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
        lm = pts[idx]
        result.append((int(lm.x * w), int(lm.y * h)))
    return result


def eye_radius(face_landmarks, frame_shape) -> int:
    h, w = frame_shape[:2]
    pts = face_landmarks.landmark
    left_outer = np.array([pts[LEFT_EYE_OUTER].x * w, pts[LEFT_EYE_OUTER].y * h], dtype=np.float32)
    left_inner = np.array([pts[LEFT_EYE_INNER].x * w, pts[LEFT_EYE_INNER].y * h], dtype=np.float32)
    eye_w = np.linalg.norm(left_outer - left_inner)
    return max(12, int(eye_w * 0.22))


# ------------------------------
# Gesture recognition
# ------------------------------
class GestureRecognizer:
    def __init__(self):
        self.history: Deque[str] = deque(maxlen=GESTURE_HISTORY)
        self.last_emit = "IDLE"
        self.last_emit_time = 0.0
        self.locked_frames = 0

    @staticmethod
    def _single_hand_label(hand: HandState) -> str:
        f = hand.finger_bits
        pinch_idx = hand.pinch_index
        pinch_mid = hand.pinch_middle

        # Improved gesture logic using fingers + distances
        if f == (0, 0, 0, 0, 0):
            return "KATON"
        if f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0:
            return "CHIDORI"
        if f == (1, 1, 1, 1, 1):
            return "RASENGAN"
        if f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 1:
            return "SHARINGAN"
        if f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 0:
            return "BIJUU_MODE"
        if f == (0, 1, 1, 1, 1):
            return "BYAKUGAN"
        if f == (0, 0, 1, 1, 1):
            return "SUITON"
        if pinch_idx < 0.35 and f[1] == 0 and sum(f[2:]) <= 1:
            return "SEAL_PINCH"
        if pinch_mid < 0.40 and f[2] == 0:
            return "CHAKRA_CHARGE"
        return "IDLE"

    @staticmethod
    def _two_hand_label(h1: HandState, h2: HandState) -> str:
        c1 = np.array(h1.center, dtype=np.float32)
        c2 = np.array(h2.center, dtype=np.float32)
        dist = np.linalg.norm(c1 - c2) / max((h1.palm_size + h2.palm_size) * 0.5, 1.0)

        if h1.finger_bits == (0, 0, 0, 0, 0) and h2.finger_bits == (0, 0, 0, 0, 0) and dist < 3.8:
            return "AMATERASU"

        vv = [(0, 1, 1, 0, 0), (1, 1, 1, 0, 0), (0, 1, 1, 0, 0), (1, 1, 0, 0, 0)]
        if h1.finger_bits in vv and h2.finger_bits in vv:
            return "SUSANOO"

        if h1.finger_bits == (1, 1, 1, 1, 1) and h2.finger_bits == (1, 1, 1, 1, 1) and dist < 5.5:
            return "ODAMA_RASENGAN"

        if h1.pinch_index < 0.35 and h2.pinch_index < 0.35:
            return "DUAL_CHAKRA"

        return "IDLE"

    def predict(self, hands: List[HandState]) -> str:
        if len(hands) >= 2:
            label = self._two_hand_label(hands[0], hands[1])
            if label == "IDLE":
                label = self._single_hand_label(hands[0])
        elif len(hands) == 1:
            label = self._single_hand_label(hands[0])
        else:
            label = "IDLE"

        self.history.append(label)
        stable = Counter(self.history).most_common(1)[0][0] if self.history else "IDLE"

        if stable != "IDLE":
            if stable == self.last_emit:
                self.locked_frames += 1
            else:
                self.last_emit = stable
                self.locked_frames = 1

            if self.locked_frames >= HOLD_FRAMES and (time.time() - self.last_emit_time) > GLOBAL_COOLDOWN:
                self.last_emit_time = time.time()
                return stable

        return "IDLE"


# ------------------------------
# FX engine
# ------------------------------
class FXEngine:
    def __init__(self):
        self.particles: List[Particle] = []
        self.anchor_histories: Dict[str, Deque[Tuple[int, int]]] = {
            "left": deque(maxlen=6),
            "right": deque(maxlen=6),
            "center": deque(maxlen=6),
        }
        self.active_jutsu = "IDLE"
        self.active_until = 0.0

    def trigger(self, jutsu: str, duration: float = 1.35):
        self.active_jutsu = jutsu
        self.active_until = time.time() + duration

    def current(self) -> str:
        if time.time() <= self.active_until:
            return self.active_jutsu
        return "IDLE"

    def emit(self, x, y, n, color, speed=(1, 5), size=(2, 7), life=(0.3, 0.9), kind="normal", spread=math.pi * 2):
        for _ in range(n):
            ang = random.uniform(0, spread)
            vel = random.uniform(*speed)
            self.particles.append(
                Particle(
                    x=float(x),
                    y=float(y),
                    vx=math.cos(ang) * vel,
                    vy=math.sin(ang) * vel,
                    life=random.uniform(*life),
                    born=time.time(),
                    size=random.uniform(*size),
                    color=color,
                    kind=kind,
                )
            )

    def update_particles(self, frame):
        self.particles = [p for p in self.particles if p.alive]
        overlay = np.zeros_like(frame)

        for p in self.particles:
            a = 1.0 - p.age_ratio
            if p.kind == "fire":
                p.vy -= 0.04
                p.vx += random.uniform(-0.15, 0.15)
            elif p.kind == "spark":
                p.vy += 0.05
            elif p.kind == "smoke":
                p.vy -= 0.02
                p.vx += random.uniform(-0.1, 0.1)
            elif p.kind == "water":
                p.vy += 0.02

            p.x += p.vx
            p.y += p.vy
            radius = max(1, int(p.size * a))
            px, py = int(p.x), int(p.y)
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                cv2.circle(overlay, (px, py), radius + 2, p.color, -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (px, py), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.22, frame, 1.0, 0, frame)

    def _anchor(self, key: str, pt: Tuple[int, int]) -> Tuple[int, int]:
        self.anchor_histories[key].append(pt)
        return smooth_point(self.anchor_histories[key])

    def render(self, frame: np.ndarray, hands: List[HandState], eye_positions: List[Tuple[int, int]], eye_r: int, t: float):
        h, w = frame.shape[:2]
        center_pt = (w // 2, h // 2)
        if hands:
            hands_sorted = sorted(hands, key=lambda x: x.center[0])
            left = hands_sorted[0]
            right = hands_sorted[-1]
            left_pt = self._anchor("left", tuple(left.pts_px[INDEX_TIP]))
            right_pt = self._anchor("right", tuple(right.pts_px[INDEX_TIP]))
            center_pt = self._anchor("center", tuple(np.mean([left.center, right.center], axis=0).astype(int)))
        else:
            left_pt = right_pt = center_pt

        jutsu = self.current()
        if jutsu == "IDLE":
            self.update_particles(frame)
            return

        put_title(frame, f"Jutsu : {jutsu}")

        if jutsu == "KATON":
            self.katon(frame, left_pt if hands else center_pt, t)
        elif jutsu == "CHIDORI":
            self.chidori(frame, left_pt if hands else center_pt, t)
        elif jutsu == "RASENGAN":
            self.rasengan(frame, left_pt if hands else center_pt, t, big=False)
        elif jutsu == "ODAMA_RASENGAN":
            self.rasengan(frame, center_pt, t, big=True)
        elif jutsu == "SHARINGAN":
            self.sharingan(frame, eye_positions, eye_r, t)
        elif jutsu == "BYAKUGAN":
            self.byakugan(frame, eye_positions, eye_r, t)
        elif jutsu == "SUSANOO":
            self.susanoo(frame, center_pt, t)
        elif jutsu == "AMATERASU":
            self.amaterasu(frame, center_pt, t)
        elif jutsu == "SUITON":
            self.suiton(frame, left_pt if hands else center_pt, t)
        elif jutsu == "BIJUU_MODE":
            self.bijuu_mode(frame, center_pt, t)
        elif jutsu == "DUAL_CHAKRA":
            self.dual_chakra(frame, left_pt, right_pt, t)
        elif jutsu == "CHAKRA_CHARGE":
            self.chakra_charge(frame, left_pt if hands else center_pt, t)
        elif jutsu == "SEAL_PINCH":
            self.seal_pinch(frame, left_pt if hands else center_pt, t)

        self.update_particles(frame)

    # ---- Individual effects ----
    def katon(self, frame, pt, t):
        x, y = pt
        draw_glow_circle(frame, (x, y), 48, (0, 140, 255))
        for _ in range(14):
            self.emit(x, y, 1, (0, random.randint(120, 180), 255), speed=(1.5, 5.5), size=(5, 11), life=(0.35, 0.8), kind="fire")
        cv2.circle(frame, (x, y), 15, (220, 255, 255), -1, lineType=cv2.LINE_AA)

    def chidori(self, frame, pt, t):
        x, y = pt
        draw_glow_circle(frame, (x, y), 56, (255, 170, 80))
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            length = random.randint(60, 180)
            end = (int(x + math.cos(angle) * length), int(y + math.sin(angle) * length))
            mid1 = (int((2 * x + end[0]) / 3 + random.randint(-18, 18)), int((2 * y + end[1]) / 3 + random.randint(-18, 18)))
            mid2 = (int((x + 2 * end[0]) / 3 + random.randint(-18, 18)), int((y + 2 * end[1]) / 3 + random.randint(-18, 18)))
            pts = [(x, y), mid1, mid2, end]
            for a, b in zip(pts[:-1], pts[1:]):
                draw_glow_line(frame, a, b, (255, 255, 255), 2)
                cv2.line(frame, a, b, (255, 230, 180), 1, lineType=cv2.LINE_AA)
        self.emit(x, y, 12, (255, 255, 255), speed=(3, 8), size=(2, 4), life=(0.2, 0.5), kind="spark")

    def rasengan(self, frame, pt, t, big=False):
        x, y = pt
        base = 90 if big else 58
        for i in range(5):
            r = int(base - i * (12 if not big else 18) + 4 * math.sin(t * 6 + i))
            draw_glow_circle(frame, (x, y), max(6, r), (255, 180, 70))
        steps = 120 if big else 80
        for i in range(steps):
            ang = t * 4 + i * 0.26
            rr = (i / steps) * base
            px = int(x + rr * math.cos(ang))
            py = int(y + rr * math.sin(ang))
            cv2.circle(frame, (px, py), 2 if not big else 3, (255, 240, 180), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), 14 if not big else 22, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        self.emit(x, y, 8 if not big else 16, (255, 220, 120), speed=(1, 3), size=(2, 5), life=(0.3, 0.7), kind="spark")

    def sharingan(self, frame, eye_positions, eye_r, t):
        if not eye_positions:
            return
        for ex, ey in eye_positions:
            draw_glow_circle(frame, (ex, ey), int(eye_r * 1.8), (0, 0, 255))
            cv2.circle(frame, (ex, ey), eye_r, (0, 0, 170), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), int(eye_r * 0.72), (0, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), int(eye_r * 0.26), (0, 0, 0), -1, lineType=cv2.LINE_AA)
            for i in range(3):
                ang = t * 4 + i * (2 * math.pi / 3)
                tx = int(ex + eye_r * 0.55 * math.cos(ang))
                ty = int(ey + eye_r * 0.55 * math.sin(ang))
                cv2.circle(frame, (tx, ty), max(2, int(eye_r * 0.12)), (0, 0, 0), -1, lineType=cv2.LINE_AA)

    def byakugan(self, frame, eye_positions, eye_r, t):
        if not eye_positions:
            return
        for ex, ey in eye_positions:
            cv2.circle(frame, (ex, ey), eye_r, (245, 245, 245), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), int(eye_r * 0.18), (220, 220, 220), -1, lineType=cv2.LINE_AA)
            for deg in range(0, 360, 18):
                ang = math.radians(deg) + t * 0.3
                length = int(eye_r * 2.9)
                p2 = (int(ex + length * math.cos(ang)), int(ey + length * math.sin(ang)))
                cv2.line(frame, (ex, ey), p2, (180, 160, 160), 1, lineType=cv2.LINE_AA)
            draw_glow_circle(frame, (ex, ey), int(eye_r * 1.7), (255, 255, 255))

    def susanoo(self, frame, pt, t):
        x, y = pt
        h, w = frame.shape[:2]
        overlay = frame.copy()
        head = (x, y - 140)
        torso_top = (x, y - 60)
        torso_bottom = (x, y + 130)
        left_arm = (x - 120, y - 20)
        right_arm = (x + 120, y - 20)
        left_hand = (x - 180, y + 70)
        right_hand = (x + 180, y + 70)
        left_leg = (x - 80, y + 260)
        right_leg = (x + 80, y + 260)
        for a, b in [
            (head, torso_top), (torso_top, torso_bottom),
            (torso_top, left_arm), (left_arm, left_hand),
            (torso_top, right_arm), (right_arm, right_hand),
            (torso_bottom, left_leg), (torso_bottom, right_leg),
        ]:
            draw_glow_line(frame, a, b, (255, 120, 255), 5)
        cv2.circle(frame, head, 44, (255, 180, 255), 3, lineType=cv2.LINE_AA)
        cv2.ellipse(overlay, (x, y + 50), (240, 290), 0, 0, 360, (120, 40, 180), -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.line(frame, (x, y - 220), (x, min(h - 1, y + 320)), (255, 0, 255), 2, lineType=cv2.LINE_AA)
        self.emit(x, y, 18, (255, 180, 255), speed=(1, 3), size=(3, 6), life=(0.3, 0.9), kind="spark")

    def amaterasu(self, frame, pt, t):
        x, y = pt
        draw_glow_circle(frame, (x, y), 100, (60, 60, 60))
        for _ in range(20):
            self.emit(x + random.randint(-30, 30), y + random.randint(-30, 30), 1, (20, 20, 20), speed=(1, 4), size=(7, 14), life=(0.5, 1.2), kind="fire")
        for i in range(4):
            rr = int(70 + 14 * math.sin(t * 5 + i))
            cv2.circle(frame, (x, y), rr, (25, 25, 25), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), 28, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def suiton(self, frame, pt, t):
        x, y = pt
        for i in range(4):
            rr = 42 + i * 16
            cv2.ellipse(frame, (x, y), (rr, int(rr * 0.55)), 0, 0, 360, (255, 150 + i * 20, 50), 2, lineType=cv2.LINE_AA)
        for i in range(8):
            ang = t * 2 + i * (2 * math.pi / 8)
            px = int(x + math.cos(ang) * 70)
            py = int(y + math.sin(ang) * 24)
            draw_glow_circle(frame, (px, py), 10, (255, 180, 60))
        self.emit(x, y, 10, (255, 180, 70), speed=(1, 4), size=(3, 6), life=(0.35, 0.8), kind="water")

    def bijuu_mode(self, frame, pt, t):
        x, y = pt
        overlay = frame.copy()
        cv2.ellipse(overlay, (x, y + 20), (180, 270), 0, 0, 360, (0, 90, 255), -1)
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
        for i in range(6):
            ang = -math.pi / 2 + (i - 2.5) * 0.32 + math.sin(t * 2 + i) * 0.08
            start = (x, y + 140)
            end = (int(x + math.cos(ang) * 230), int(y + 140 + math.sin(ang) * 130))
            draw_glow_line(frame, start, end, (0, 160, 255), 7)
        cv2.circle(frame, (x, y - 40), 60, (0, 180, 255), 3, lineType=cv2.LINE_AA)
        self.emit(x, y, 16, (80, 200, 255), speed=(1, 4), size=(3, 7), life=(0.4, 1.0), kind="spark")

    def dual_chakra(self, frame, left_pt, right_pt, t):
        draw_glow_circle(frame, left_pt, 36, (255, 180, 80))
        draw_glow_circle(frame, right_pt, 36, (255, 180, 80))
        draw_glow_line(frame, left_pt, right_pt, (255, 230, 180), 4)
        self.emit(*left_pt, 5, (255, 220, 120), speed=(1, 3), size=(2, 5), life=(0.2, 0.5), kind="spark")
        self.emit(*right_pt, 5, (255, 220, 120), speed=(1, 3), size=(2, 5), life=(0.2, 0.5), kind="spark")

    def chakra_charge(self, frame, pt, t):
        x, y = pt
        for i in range(6):
            ang = t * 3 + i * (2 * math.pi / 6)
            px = int(x + math.cos(ang) * 42)
            py = int(y + math.sin(ang) * 42)
            draw_glow_circle(frame, (px, py), 12, (255, 200, 100))
        draw_glow_circle(frame, (x, y), 28, (255, 220, 150))
        self.emit(x, y, 12, (255, 220, 150), speed=(1, 4), size=(2, 5), life=(0.25, 0.7), kind="spark")

    def seal_pinch(self, frame, pt, t):
        x, y = pt
        pts = []
        r = 54
        for i in range(6):
            ang = t * 0.8 + i * (2 * math.pi / 6)
            pts.append((int(x + r * math.cos(ang)), int(y + r * math.sin(ang))))
        for a, b in zip(pts, pts[1:] + pts[:1]):
            draw_glow_line(frame, a, b, (255, 255, 255), 2)
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (255, 220, 150), 1, lineType=cv2.LINE_AA)


# ------------------------------
# Main app
# ------------------------------
def run():
    draw_landmarks = DRAW_LANDMARKS
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX or camera permissions.")

    writer = None
    if ENABLE_RECORDING:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20.0, (FRAME_W, FRAME_H))

    recognizer = GestureRecognizer()
    fx = FXEngine()

    prev = time.time()

    with mp_hands.Hands(
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=0.72,
        min_tracking_confidence=0.68,
    ) as hands_model, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    ) as face_model:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hands_result = hands_model.process(rgb)
            face_result = face_model.process(rgb)
            rgb.flags.writeable = True

            hand_states: List[HandState] = []
            eye_positions: List[Tuple[int, int]] = []
            eye_r = 18

            if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
                pairs = list(zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness))
                for hand_landmarks, handedness in pairs:
                    st = build_hand_state(hand_landmarks, handedness, frame.shape)
                    hand_states.append(st)
                    if draw_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_draw_styles.get_default_hand_landmarks_style(),
                            mp_draw_styles.get_default_hand_connections_style(),
                        )
                    x1, y1, x2, y2 = st.bbox
                    cv2.rectangle(frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (60, 180, 255), 1)
                    cv2.putText(frame, f"{st.label} {st.finger_bits}", (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            if face_result.multi_face_landmarks:
                face_landmarks = face_result.multi_face_landmarks[0]
                eye_positions = extract_eye_positions(face_landmarks, frame.shape)
                eye_r = eye_radius(face_landmarks, frame.shape)

            predicted = recognizer.predict(hand_states)
            if predicted != "IDLE":
                fx.trigger(predicted)

            t = time.time()
            fx.render(frame, hand_states, eye_positions, eye_r, t)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            draw_fps(frame, fps)
            put_subtitle(frame, "Controls: Q quit | R screenshot | V landmarks on/off", 74)
            put_subtitle(frame, "Better than basic demos: temporal smoothing, cooldowns, eye FX, modular jutsu engine", 98, (120, 255, 255), 0.5)

            cv2.imshow(WINDOW_NAME, frame)
            if writer is not None:
                writer.write(cv2.resize(frame, (FRAME_W, FRAME_H)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                draw_landmarks = not draw_landmarks
            elif key == ord('r'):
                cv2.imwrite(f"screenshot_{int(time.time())}.png", frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
