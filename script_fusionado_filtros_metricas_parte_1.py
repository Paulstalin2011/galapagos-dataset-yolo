
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Rutas base
base_path = r"C:\Users\Paul Stalin\Documents\PaulDCC"
foto_path = os.path.join(base_path, "Foto.jpg")
output_dir = os.path.join(base_path, "reporte_datos")
os.makedirs(output_dir, exist_ok=True)

# Cargar imagen
imagen = cv2.imread(foto_path)
if imagen is None:
    raise FileNotFoundError("No se pudo cargar Foto.jpg")

# Filtros
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

# Diccionario de filtros
filtros = {
    "original": imagen,
    "clahe": aplicar_clahe(imagen),
    "canny": aplicar_canny(imagen),
    "blur": aplicar_blur(imagen),
    "sharpen": aplicar_sharpen(imagen),
    "equalize": aplicar_equalize(imagen),
    "laplacian": aplicar_laplacian(imagen)
}

metricas = []
for nombre, img in filtros.items():
    path = os.path.join(output_dir, f"{nombre}.jpg")
    cv2.imwrite(path, img)
    ent, var = calcular_metrica(img)
    metricas.append((nombre, ent, var))

# Guardar las métricas en .txt
txt_path = os.path.join(output_dir, "metricas_filtros.txt")
with open(txt_path, "w") as f:
    for nombre, ent, var in metricas:
        f.write(f"{nombre.upper()}\nEntropía: {ent}\nVarianza: {var}\n\n")

# Crear gráficas
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

print("[✅] Filtros, métricas y gráficas generadas en:", output_dir)
