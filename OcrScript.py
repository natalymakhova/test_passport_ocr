import cv2
from re import sub
import easyocr
import imutils
import numpy as np
from skimage.filters import unsharp_mask


# Добавление рамки к изображению, дающие возможность расширять mrz
def makeBord(image, bordersize=10):
    row, col = image.shape[:2]
    bottom = image[row - 2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    border = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )
    return border


# Упорядочивание контуров по оси y для основного текста, и оси x для серии и номера
def sort_contours(cnts, method):
    reverse = False
    i = 0
    if method == "top-to-bottom":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return boundingBoxes


# Поиск и выделение машиночитаемых зон
def findMRZ(image, method):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, sqKernel)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    boundingBoxes = sort_contours(cnts, method)

    return (cnts, boundingBoxes)


def prepare_im(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    unsh = unsharp_mask(gray, radius=5.3, amount=2.6)
    prep_im = (unsh * 255).astype(np.uint8)

    return prep_im

# Распознавание текста в машиночитаемых зонах
def get_text(image_initial, im_prep, horisontal=False):
    text = []
    method = 'top-to-bottom'

    if horisontal:
        image_initial = cv2.rotate(image_initial, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im_prep = cv2.rotate(im_prep, cv2.ROTATE_90_COUNTERCLOCKWISE)
        method = 'left-to-right'

    (cnts, boundingBoxes) = findMRZ(image_initial, method)
   # По порядку обрабатываем фрагменты изображения, в которых содержится текст
    for j in range(len(boundingBoxes)):
        (x, y, w, h) = boundingBoxes[j]
        #  Проверка размеров фрагмента
        if w > 20 and h > 10 and h < 30:
            # cv2.imshow('fragm', im_prep[y - 5:y + 5 + h, x - 5:x + 5 + w].copy())
            roi = im_prep[y - 5:y + 5 + h, x - 5:x + 5 + w].copy()
            roi = cv2.resize(roi, None, fx=3.5, fy=3, interpolation=cv2.INTER_CUBIC)
            words = reader.readtext(roi, detail=0)
            if words:
                text += [sub("[^А-Я 0-9.-]", "", word.upper()) for word in words]


    return text


reader = easyocr.Reader(['ru', 'en'])


def get_document_data(image_path):
    image_initial = cv2.imread(image_path)
    image = imutils.resize(image_initial, height=800)
    image = makeBord(image)
    # Предварительная обработка изображения для распознавания символов
    im_prep = prepare_im(image)

    # Основной текст
    main_text = get_text(image, im_prep, horisontal=False)

    # Серия и номер
    doc_number = get_text(image, im_prep, horisontal=True)

    return main_text + doc_number

