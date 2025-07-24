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

    selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)
    seg_res = selfie_seg.process(rgb)
    mask_person = (seg_res.segmentation_mask > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_person = cv2.morphologyEx(mask_person, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_person = cv2.morphologyEx(mask_person, cv2.MORPH_OPEN, kernel, iterations=1)
    img_clean = cv2.bitwise_and(img, img, mask=mask_person)
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(blur, 30, 120)
    skeleton = thinning(edges)
    skeleton = cv2.dilate(skeleton, np.ones((2, 2), np.uint8), iterations=1)
    
    cv2.imshow("Original", img)
    cv2.imshow("Mask Person", mask_person * 255)
    cv2.imshow("Cleaned Image", img_clean)
    cv2.imshow("Grayscale", blur)
    cv2.imshow("Canny Edges", edges)
    cv2.imshow("Skeleton", skeleton)
    #cv2.imshow("Paso 7 - Canvas Final", canvas)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #LANDMARKS   

    canvas = np.ones((h, w, 3), np.uint8) * 255
    contours = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        if cv2.contourArea(c) < 30: continue
        poly = cv2.approxPolyDP(c, 2, True)
        cv2.polylines(canvas, [poly], True, (0, 0, 0), 1)

    canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    return canvas, skeleton

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

def dibujar_con_dobot(api, skeleton):
    regiones = []
    for c in cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) < 10: continue
        regiones.append([tuple(p[0]) for p in cv2.approxPolyDP(c, 2, True)])

    for poly in regiones:
        smooth = suavizar_curva(poly)
        pts_d = [(int(x_init + escala * x), int(y_init - escala * y)) for x, y in smooth]
        draw_polyline(api, pts_d)

    dType.DisconnectDobot(api)
    print("Trazado completado.")

# === EJECUCIÓN ===

if __name__ == "__main__":
    ruta_imagen = "person5.jpg"
    imagen = cargar_imagen_desde_archivo(ruta_imagen)
    canvas, skeleton = procesar_imagen(imagen)
    mostrar_preview(canvas)
    robot = conectar_dobot()
    dibujar_con_dobot(robot, skeleton)
