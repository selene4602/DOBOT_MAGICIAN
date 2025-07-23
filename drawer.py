# Requisitos previos:
# pip install opencv-python mediapipe numpy scipy opencv-contrib-python DobotDllType
# mediapipe 64bits, python version <3.10
import os
os.environ['GLOG_minloglevel'] = '2'
from absl import logging as absl_logging
#absl_logging.set_verbosity(absl_logging.ERROR)

import cv2
import numpy as np
import mediapipe as mp
import cv2.ximgproc as xip
import DobotDllType as dType
import time
from scipy.interpolate import splprep, splev

x_init, y_init, z_up, z_down = 170, 10, -10, -23  
escala = 0.5
vel = 100         
sleep_draw = 0.005  

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_selfie = mp.solutions.selfie_segmentation
selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)

def draw_polyline(points):
    if len(points) < 2:
        return
    x0, y0 = points[0]
    dType.SetPTPCmdEx(api, dType.PTPMode.PTPMOVJXYZMode, x0, y0, z_up, 0, 1)
    time.sleep(sleep_draw)
    dType.SetPTPCmdEx(api, dType.PTPMode.PTPMOVLXYZMode, x0, y0, z_down, 0, 1)
    time.sleep(sleep_draw)
    for (x, y) in points[1:]:
        dType.SetPTPCmdEx(api, dType.PTPMode.PTPMOVLXYZMode, x, y, z_down, 0, 1)
        time.sleep(sleep_draw)
    dType.SetPTPCmdEx(api, dType.PTPMode.PTPMOVJXYZMode, points[-1][0], points[-1][1], z_up, 0, 1)
    time.sleep(sleep_draw)


def suavizar_curva(puntos, resolucion=15):
    if len(puntos) < 3:
        return puntos
    pts = np.array(puntos)
    s = len(pts) / 20
    try:
        pts_ext = np.vstack([[pts[0]], pts, [pts[-1]]])
        tck, u = splprep(pts_ext.T, s=s)
        u_new = np.linspace(0, 1, resolucion)
        xs, ys = splev(u_new, tck)
        return list(zip(xs, ys))
    except:
        return puntos

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()
print("Presiona 's' para capturar la imagen, ESC para salir.")
img_color = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_color = frame.copy()
        break
    elif key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()
cap.release()
cv2.destroyAllWindows()
if img_color is None:
    print("No se capturó ninguna imagen. Saliendo...")
    exit()

h, w = 200,200
img = cv2.resize(img_color, (w, h))
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Segmentar persona vs fondo ---
seg_res = selfie_seg.process(rgb)
mask_person = (seg_res.segmentation_mask > 0.5).astype(np.uint8)
# Refina máscara
kernel = np.ones((5,5), np.uint8)
mask_person = cv2.morphologyEx(mask_person, cv2.MORPH_CLOSE, kernel, iterations=2)

# --- Contorno general (solo persona) ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 7, 50, 50)
edges = cv2.Canny(blur, 30, 120)
edges = cv2.bitwise_and(edges, edges, mask=mask_person)
skeleton = xip.thinning(edges)

results = face_mesh.process(rgb)
if not results.multi_face_landmarks:
    print("No se detectó rostro.")
    exit()
landmarks = results.multi_face_landmarks[0]
pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]


# --- Máscara de cabello (tonos oscuros en zona superior) ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_hair = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
mask_hair[:int(h*0.3), :] = 0  # omite parte muy superior si sobra
hair_skel = xip.thinning(mask_hair)

# --- Máscara de barba (tonos oscuros debajo de labios) ---
w_lower = [61, 291]  # índices aproximados de labios inferior
y_lip = int(max(pts[i][1] for i in w_lower))
mask_beard = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
mask_beard = cv2.bitwise_and(mask_beard, mask_beard, mask=mask_person)
mask_beard[:y_lip, :] = 0
beard_skel = xip.thinning(mask_beard)

canvas = np.ones((h, w, 3), np.uint8)*255
#canvas[skeleton > 0] = (0,0,0)
#canvas[hair_skel > 0] = (0,0,0)
#canvas[beard_skel > 0] = (0,0,0)

for m in [hair_skel, beard_skel]:
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for  c in cnts:
        poly = cv2.approxPolyDP(c, 2, True)
        cv2.polylines(canvas, [poly], True, (0,0,0), 1)
# Dibujar detalles internos: labios, ojos, cejas

feature_groups = [
    #mp_face_mesh.FACEMESH_FACE_OVAL,
    mp_face_mesh.FACEMESH_LIPS,
    mp_face_mesh.FACEMESH_LEFT_EYE, 
    mp_face_mesh.FACEMESH_RIGHT_EYE,
    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW
]

for grp in feature_groups:
    seq = []  # mod at 111
    for s, e in grp:
        if pts[s] not in seq: seq.append(pts[s])  # mod at 112: unique
        if pts[e] not in seq: seq.append(pts[e])  # mod at 113: unique
    cv2.polylines(canvas, [np.array(seq)], False, (0,0,0), 1)  # mod at 114: one polyline per feature
# Rotar preview
        
canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
while True:
    cv2.imshow('Preview Final Rotado', canvas)
    k=cv2.waitKey(1)
    if k==ord('s'):
        cv2.imwrite('preview_saved.png', canvas)
        print('Guardado como preview_saved.png')
    elif k==27:
        break
cv2.destroyAllWindows()

com_port = "COM6"
print(f"Conectando a Dobot en {com_port}...")
api = dType.load()
state = dType.ConnectDobot(api, com_port, 115200)[0]
if state != dType.DobotConnect.DobotConnect_NoError:
    print("Error al conectar con Dobot en COM8", state)
    exit()
dType.SetQueuedCmdClear(api)
time.sleep(1)
dType.SetPTPCommonParams(api, vel, vel)

regions = []

for m in [hair_skel, beard_skel]:  # línea 131
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:  # mod at 132
        regions.append([tuple(p[0]) for p in cv2.approxPolyDP(c, 2, True)])
for grp in feature_groups:  # línea 135
    # seq construida igual al preview
    seq = []  # mod at 136
    for s, e in grp:
        if pts[s] not in seq: seq.append(pts[s])  # mod at 137
        if pts[e] not in seq: seq.append(pts[e])  # mod at 138
    regions.append(seq)

for poly in regions:
    smooth=suavizar_curva(poly)
    pts_d=[(int(x_init+escala*x),int(y_init-escala*y)) for x,y in smooth]
    draw_polyline(pts_d)

DobotDisconnected = dType.DisconnectDobot(api)
print("Trazado optimizado completado en COM8.")
