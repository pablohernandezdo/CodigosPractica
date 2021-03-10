import os
import cv2


def main():
    # Carpeta para guardar las fotos procesadas
    os.makedirs('Estimaciones', exist_ok=True)

    # Leer originales
    calle1030orig = cv2.imread("FotosTemp/calle1030original.jpg")
    calle1070orig = cv2.imread("FotosTemp/calle1070original.jpg")
    silla1030orig = cv2.imread("FotosTemp/silla1030original.jpg")
    silla1070orig = cv2.imread("FotosTemp/silla1070original.jpg")

    # Leer muestras imágenes de muestra para leer posicion de los
    # puntos. No se usa para estimar.

    calle1030muestra1 = cv2.imread("FotosTemp/calle1030muestra1.jpg")
    calle1030muestra2 = cv2.imread("FotosTemp/calle1030muestra2.jpg")
    calle1030muestra3 = cv2.imread("FotosTemp/calle1030muestra3.jpg")

    calle1070muestra1 = cv2.imread("FotosTemp/calle1070muestra1.jpg")
    calle1070muestra2 = cv2.imread("FotosTemp/calle1070muestra2.jpg")
    calle1070muestra3 = cv2.imread("FotosTemp/calle1070muestra3.jpg")

    silla1030muestra1 = cv2.imread("FotosTemp/silla1030muestra1.jpg")
    silla1030muestra2 = cv2.imread("FotosTemp/silla1030muestra2.jpg")
    silla1030muestra3 = cv2.imread("FotosTemp/silla1030muestra3.jpg")

    silla1070muestra1 = cv2.imread("FotosTemp/silla1070muestra1.jpg")
    silla1070muestra2 = cv2.imread("FotosTemp/silla1070muestra2.jpg")
    silla1070muestra3 = cv2.imread("FotosTemp/silla1070muestra3.jpg")

    # Posición aproximada de las muestras obtenida manualmente
    calle1030muestras_pos = [(311, 234), (155, 195), (425, 460)]
    calle1070muestras_pos = [(160, 380), (450, 225), (265, 160)]
    silla1030muestras_pos = [(482, 327), (120, 298), (543, 447)]
    silla1070muestras_pos = [(477, 300), (103, 221), (436, 119)]

    # Valores muestras
    calle1030muestras_temp = [19.9, 16.5, 15.1]
    calle1070muestras_temp = [20.8, 15.6, 19.5]
    silla1030muestras_temp = [17.6, 18.1, 23.1]
    silla1070muestras_temp = [21.5, 14.9, 20.2]

    # Imagen escala de grises
    calle1030orig_gray = cv2.cvtColor(calle1030orig, cv2.COLOR_BGR2GRAY)
    calle1070orig_gray = cv2.cvtColor(calle1070orig, cv2.COLOR_BGR2GRAY)
    silla1030orig_gray = cv2.cvtColor(silla1030orig, cv2.COLOR_BGR2GRAY)
    silla1070orig_gray = cv2.cvtColor(silla1070orig, cv2.COLOR_BGR2GRAY)

    # Transformación de luminancia
    calle1030orig_lum = luminance_transform(calle1030orig)
    calle1070orig_lum = luminance_transform(calle1070orig)
    silla1030orig_lum = luminance_transform(silla1030orig)
    silla1070orig_lum = luminance_transform(silla1070orig)

    # Interpolaciones
    calle1030temps_gray = interpolate_temp(calle1030orig_gray, 10, 30)
    calle1070temps_gray = interpolate_temp(calle1070orig_gray, 10, 70)
    silla1030temps_gray = interpolate_temp(silla1030orig_gray, 10, 30)
    silla1070temps_gray = interpolate_temp(silla1070orig_gray, 10, 70)

    calle1030temps_lum = (30-10) * calle1030orig_lum.astype('double') / 255 + 10
    calle1070temps_lum = (70-10) * calle1070orig_lum.astype('double') / 255 + 10
    silla1030temps_lum = (30-10) * silla1030orig_lum.astype('double') / 255 + 10
    silla1070temps_lum = (70-10) * silla1070orig_lum.astype('double') / 255 + 10

    calle1030temps_lum = interpolate_temp(calle1030orig_lum, 10, 30)
    calle1070temps_lum = interpolate_temp(calle1070orig_lum, 10, 70)
    silla1030temps_lum = interpolate_temp(silla1030orig_lum, 10, 30)
    silla1070temps_lum = interpolate_temp(silla1070orig_lum, 10, 70)

    # Dibujar rectangulos en las imagenes y añadir temperatura estimada

    # Parametros cuadro de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 1

    print(f'CALLE1030\n')

    for pos, temp in zip(calle1030muestras_pos, calle1030muestras_temp):
        cv2.rectangle(calle1030orig,
                      (pos[0] - 5, pos[1] - 5),
                      (pos[0] + 5, pos[1] + 5),
                      (255, 255, 255), 2)

        cv2.putText(calle1030orig,
                    f'({calle1030temps_gray[pos[1], pos[0]]:5.3f}, {calle1030temps_lum[pos[1], pos[0]]:5.3f}, {temp})',
                    (pos[0], pos[1] - 10),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    print(f'CALLE1070\n')

    for pos, temp in zip(calle1070muestras_pos, calle1070muestras_temp):
        cv2.rectangle(calle1070orig,
                      (pos[0] - 5, pos[1] - 5),
                      (pos[0] + 5, pos[1] + 5),
                      (255, 255, 255), 2)

        cv2.putText(calle1070orig,
                    f'({calle1070temps_gray[pos[1], pos[0]]:5.3f}, {calle1070temps_lum[pos[1], pos[0]]:5.3f}, {temp})',
                    (pos[0], pos[1] - 10),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    print(f'SILLA1030\n')

    for pos, temp in zip(silla1030muestras_pos, silla1030muestras_temp):
        cv2.rectangle(silla1030orig,
                      (pos[0] - 5, pos[1] - 5),
                      (pos[0] + 5, pos[1] + 5),
                      (255, 255, 255), 2)

        cv2.putText(silla1030orig,
                    f'({silla1030temps_gray[pos[1], pos[0]]:5.3f}, {silla1030temps_lum[pos[1], pos[0]]:5.3f}, {temp})',
                    (pos[0] - 100, pos[1] - 10),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    print(f'SILLA1070\n')

    for pos, temp in zip(silla1070muestras_pos, silla1070muestras_temp):
        cv2.rectangle(silla1070orig,
                      (pos[0] - 5, pos[1] - 5),
                      (pos[0] + 5, pos[1] + 5),
                      (255, 255, 255), 2)

        cv2.putText(silla1070orig,
                    f'({silla1070temps_gray[pos[1], pos[0]]:5.3f}, {silla1070temps_lum[pos[1], pos[0]]:5.3f}, {temp})',
                    (pos[0] - 100, pos[1] - 10),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    # Guardar estimaciones
    cv2.imwrite('Estimaciones/Est_Calle1030.png', calle1030orig)
    cv2.imwrite('Estimaciones/Est_Calle1070.png', calle1070orig)
    cv2.imwrite('Estimaciones/Est_Silla1030.png', silla1030orig)
    cv2.imwrite('Estimaciones/Est_Silla1070.png', silla1070orig)


def luminance_transform(image):

    return (0.212 * image[:, :, 2] +
            0.7152 * image[:, :, 1] +
            0.0722 * image[:, :, 0]).astype('uint8')


def interpolate_temp(gray_image, low_temp, high_temp):
    temp_range = high_temp - low_temp
    return temp_range * gray_image.astype('double') / 255 + low_temp


if __name__ == "__main__":
    main()
