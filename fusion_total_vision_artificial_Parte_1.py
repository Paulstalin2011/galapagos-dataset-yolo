
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# CONFIGURACIÓN DE RUTAS
base_path = r"C:\Users\Paul Stalin\Documents\PaulDCC"
foto_path = os.path.join(base_path, "Foto.jpg")
textura_path = os.path.join(base_path, "Imagen.jpg")
cuadro_path = os.path.join(base_path, "Cuadro.jpg")
video_path = os.path.join(base_path, "Video.mp4")
output_dir = os.path.join(base_path, "reporte_datos")
os.makedirs(output_dir, exist_ok=True)

# FUNCIONES DE FILTROS
def aplicar_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def aplicar_canny(img):
    return cv2.cvtColor(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR)

def aplicar_blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)

def aplicar_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def aplicar_equalize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def aplicar_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.cvtColor(np.uint8(np.absolute(lap)), cv2.COLOR_GRAY2BGR)

def calcular_metrica(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist /= hist.sum()
    entropia = -np.sum(hist * np.log2(hist + 1e-7))
    varianza = np.var(gray)
    return round(entropia, 3), round(varianza, 3)

# ZONAS POLIGONALES
zonas_poligonales = []
zona_actual = []
modo_fusion = "normal"
video_zona_index = -1

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        zona_actual.append((x, y))

def mostrar_zonas(img):
    temp = img.copy()
    for zona in zonas_poligonales:
        pts = np.array(zona, np.int32).reshape((-1, 1, 2))
        cv2.polylines(temp, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    if len(zona_actual) > 1:
        pts = np.array(zona_actual, np.int32).reshape((-1, 1, 2))
        cv2.polylines(temp, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
    return temp

def fusionar(persona, textura, shape, modo):
    persona_resized = cv2.resize(persona, shape)
    textura_resized = cv2.resize(textura, shape)

    if modo == "canny":
        return aplicar_canny(persona_resized)
    elif modo == "clahe":
        return aplicar_clahe(persona_resized)
    else:
        return cv2.addWeighted(persona_resized, 0.5, textura_resized, 0.5, 0)

def aplicar_fusiones(img, zonas, persona, textura, video_frame=None):
    resultado = img.copy()
    for i, zona in enumerate(zonas):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts = np.array(zona, np.int32)
        cv2.fillPoly(mask, [pts], 255)
        x, y, w, h = cv2.boundingRect(pts)
        if i == video_zona_index and video_frame is not None:
            contenido = cv2.resize(video_frame, (w, h))
        else:
            contenido = fusionar(persona, textura, (w, h), modo_fusion)
        contenido_masked = cv2.bitwise_and(contenido, contenido, mask=mask[y:y+h, x:x+w])
        roi = resultado[y:y+h, x:x+w]
        roi_masked = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask[y:y+h, x:x+w]))
        resultado[y:y+h, x:x+w] = cv2.add(roi_masked, contenido_masked)
    return resultado

# CARGA DE IMÁGENES
cuadro_img = cv2.imread(cuadro_path)
persona = cv2.imread(foto_path)
textura = cv2.imread(textura_path)

if cuadro_img is None or persona is None or textura is None:
    raise FileNotFoundError("No se pudo cargar una de las imágenes base.")

# SELECCIÓN DE ZONAS
cv2.namedWindow("Zonas")
cv2.setMouseCallback("Zonas", mouse_callback)

while True:
    display = mostrar_zonas(cuadro_img)
    cv2.imshow("Zonas", display)
    key = cv2.waitKey(1) & 0xFF
    if key == 13 and len(zona_actual) >= 3:
        zonas_poligonales.append(zona_actual.copy())
        zona_actual.clear()
    elif key == ord('c'):
        modo_fusion = "canny"
    elif key == ord('f'):
        modo_fusion = "clahe"
    elif key == ord('n'):
        modo_fusion = "normal"
    elif key == 27:
        break

cv2.destroyAllWindows()
if not zonas_poligonales:
    raise ValueError("No se seleccionaron zonas.")
video_zona_index = len(zonas_poligonales) - 1

# INICIO DE VIDEO
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    resultado = aplicar_fusiones(cuadro_img, zonas_poligonales, persona, textura, video_frame=frame)
    fps = int(1 / max((time.time() - start_time), 0.001))
    cv2.putText(resultado, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Vista Optimizada", resultado)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite(os.path.join(output_dir, "resultado_final_optimo.jpg"), resultado)
        # Generar filtros para Foto.jpg
        filtros = {
            "original": persona,
            "clahe": aplicar_clahe(persona),
            "canny": aplicar_canny(persona),
            "blur": aplicar_blur(persona),
            "sharpen": aplicar_sharpen(persona),
            "equalize": aplicar_equalize(persona),
            "laplacian": aplicar_laplacian(persona)
        }
        metricas = []
        for nombre, img in filtros.items():
            cv2.imwrite(os.path.join(output_dir, f"{nombre}.jpg"), img)
            ent, var = calcular_metrica(img)
            metricas.append((nombre, ent, var))
        with open(os.path.join(output_dir, "metricas_filtros.txt"), "w") as f:
            for nombre, ent, var in metricas:
                f.write(f"{nombre.upper()}\nEntropía: {ent}\nVarianza: {var}\n\n")
        filtros_nombres = [m[0] for m in metricas]
        entropias = [m[1] for m in metricas]
        varianzas = [m[2] for m in metricas]
        plt.figure(figsize=(10,5))
        plt.bar(filtros_nombres, entropias, color='skyblue')
        plt.title("Entropía por Filtro")
        plt.ylabel("Entropía")
        plt.savefig(os.path.join(output_dir, "grafica_entropia.png"))
        plt.close()
        plt.figure(figsize=(10,5))
        plt.bar(filtros_nombres, varianzas, color='lightcoral')
        plt.title("Varianza por Filtro")
        plt.ylabel("Varianza")
        plt.savefig(os.path.join(output_dir, "grafica_varianza.png"))
        plt.close()
        print("[💾] Imágenes y métricas guardadas en:", output_dir)

cap.release()
cv2.destroyAllWindows()
