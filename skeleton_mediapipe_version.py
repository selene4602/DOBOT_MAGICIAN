# Requisitos previos:
# pip install opencv-python mediapipe numpy scipy opencv-contrib-python DobotDllType
# mediapipe 64bits, python version <3.10

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from cv2.ximgproc import thinning
from scipy.interpolate import splprep, splev
import DobotDllType as dType

# Parámetros Dobot
x_init, y_init, z_up, z_down = 170, 10, -10, -23
escala = 0.5
vel = 100
sleep_draw = 0.005
com_port = "COM6"

mp_face_mesh = mp.solutions.face_mesh
mp_selfie = mp.solutions.selfie_segmentation

def cargar_imagen_desde_archivo(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise Exception(f"No se pudo cargar la imagen desde {ruta_imagen}")
    return img
"""
def capturar_imagen():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo abrir la cámara.")

    print("Presiona 's' para capturar la imagen, ESC para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()
"""

def suavizar_curva(puntos, resolucion=15):
    if len(puntos) < 3:
        return puntos
    pts = np.array(puntos)
    s = max(len(pts) / 25, 1.0)
    try:
        pts_ext = np.vstack([[pts[0]], pts, [pts[-1]]])
        tck, _ = splprep(pts_ext.T, s=s)
        u_new = np.linspace(0, 1, resolucion)
        xs, ys = splev(u_new, tck)
        return list(zip(xs, ys))
    except:
        return puntos

def construir_secuencia(grupo, pts):
    seq = []
    for s, e in grupo:
        for i in (s, e):
            p = pts[i]
            if p not in seq:
                seq.append(p)
    return seq

def extraer_contornos(mascara):
    cnts = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return [tuple(p[0]) for c in cnts for p in cv2.approxPolyDP(c, 2, True)]

def procesar_imagen(img_color, w=200, h=200):
    img = cv2.resize(img_color, (w, h))
    #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow("Imagen rotada y redimensionada", img)
    cv2.imwrite("debug_1_redimensionada.jpg", img)

    selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)
    seg_res = selfie_seg.process(rgb)
    mask_person = (seg_res.segmentation_mask > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_person = cv2.morphologyEx(mask_person, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_person = cv2.morphologyEx(mask_person, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("Mascara de persona", mask_person * 255)
    cv2.imwrite("debug_2_mascara_persona.jpg", mask_person * 255)
    img_clean = cv2.bitwise_and(img, img, mask=mask_person)
    cv2.imshow("Imagen sin fondo", img_clean)
    cv2.imwrite("debug_3_img_sin_fondo.jpg", img_clean)
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 50, 50)
    cv2.imshow("Grises + bilateral", blur)
    cv2.imwrite("debug_4_grises_bilateral.jpg", blur)
    edges = cv2.Canny(blur, 30, 120)
    cv2.imshow("Canny", edges)
    cv2.imwrite("debug_5_canny.jpg", edges)
    skeleton = thinning(edges)
    skeleton = cv2.dilate(skeleton, np.ones((2, 2), np.uint8), iterations=1)
    cv2.imshow("Skeleton dilatado", skeleton)
    cv2.imwrite("debug_6_skeleton_dilatado.jpg", skeleton)

    #LANDMARKS    
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        raise Exception("No se detectó rostro.")

    landmarks = results.multi_face_landmarks[0]

    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
    for x, y in pts:
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    cv2.imshow("Landmarks", img)
    cv2.imwrite("debug_7_landmarks.jpg", img)

    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask_dark = cv2.inRange(v, 0, 60)
    y_ojos = int(np.mean([pts[33][1], pts[263][1]]))
    mask_dark[y_ojos:, :] = 0
    mask_dark = cv2.GaussianBlur(mask_dark, (5, 5), 0)
    hair_skel = thinning(mask_dark)
    hair_skel = cv2.dilate(hair_skel, np.ones((2, 2), np.uint8), iterations=2)
    #hair_skel = cv2.dilate(hair_skel, np.ones((2, 2), np.uint8), iterations=1)
    cv2.imshow("Hair Skeleton", hair_skel)
    cv2.imwrite("debug_8_skeleton_hair.jpg", hair_skel)

    
    mask_beard = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
    mask_beard = cv2.bitwise_and(mask_beard, mask_beard, mask=mask_person)
    y_lip = int(max(pts[i][1] for i in [61, 291]))
    mask_beard[:y_lip, :] = 0
    beard_skel = thinning(mask_beard)
    beard_skel = cv2.dilate(beard_skel, np.ones((2, 2), np.uint8), iterations=1)
    cv2.imshow("Beard Skeleton", beard_skel)
    cv2.imwrite("debug_7_skeleton_beard.jpg", beard_skel)
    """
    feature_groups = [
        #mp_face_mesh.FACEMESH_FACE_OVAL,
        mp_face_mesh.FACEMESH_LIPS,
        mp_face_mesh.FACEMESH_LEFT_EYE,
        mp_face_mesh.FACEMESH_RIGHT_EYE,
        mp_face_mesh.FACEMESH_LEFT_EYEBROW,
        mp_face_mesh.FACEMESH_RIGHT_EYEBROW
    ]

    canvas = np.ones((h, w, 3), np.uint8) * 255

    """for mask in [hair_skel, beard_skel]:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in contours:
            if cv2.contourArea(c) < 30: continue
            poly = cv2.approxPolyDP(c, 2, True)
            cv2.polylines(canvas, [poly], True, (0, 0, 0), 1)
    """
    for grp in feature_groups:
        seq = []  
        for s, e in grp:
            if pts[s] not in seq: seq.append(pts[s])  
            if pts[e] not in seq: seq.append(pts[e])  
        cv2.polylines(canvas, [np.array(seq)], False, (0,0,0), 1)

    canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    return canvas, skeleton, feature_groups, pts

def mostrar_preview(canvas):
    while True:
        cv2.imshow('Vista previa', canvas)
        k = cv2.waitKey(1)
        if k == ord('s'):
            cv2.imwrite('preview_saved.png', canvas)
            print('Guardado como preview_saved.png')
        elif k == 27:
            break
    cv2.destroyAllWindows()

def conectar_dobot():
    print(f"Conectando a Dobot en {com_port}...")
    api = dType.load()
    state = dType.ConnectDobot(api, com_port, 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        raise Exception(f"No se pudo conectar al Dobot en {com_port}")
    dType.SetQueuedCmdClear(api)
    time.sleep(1)
    dType.SetPTPCommonParams(api, vel, vel)
    return api

def draw_polyline(api, points):
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

def dibujar_con_dobot(api, skeleton, feature_groups, pts):
    regiones = []
    for grp in feature_groups:
            seq = []  # mod at 136
    for s, e in grp:
        if pts[s] not in seq: seq.append(pts[s])  # mod at 137
        if pts[e] not in seq: seq.append(pts[e])  # mod at 138
    regiones.append(seq)

    for poly in regiones:
        smooth = suavizar_curva(poly)
        pts_d = [(int(x_init + escala * x), int(y_init - escala * y)) for x, y in smooth]
        draw_polyline(api, pts_d)

    dType.DisconnectDobot(api)
    print("Trazado completado.")

# === EJECUCIÓN ===

if __name__ == "__main__":
    ruta_imagen = "person3.jpg"
    imagen = cargar_imagen_desde_archivo(ruta_imagen)
    canvas, skeleton, grupos, puntos = procesar_imagen(imagen)
    mostrar_preview(canvas)
    robot = conectar_dobot()
    dibujar_con_dobot(robot, skeleton, grupos, puntos)

