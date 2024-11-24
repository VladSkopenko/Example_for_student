import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def preprocess_image(image,  target_size=(32, 32)):
    """
    Змінює розмір зображення та нормалізує його.

    Args:
        image (PIL.Image.Image): Зображення для обробки.
        target_size (tuple): Бажаний розмір (ширина, висота).

    Returns:
        np.ndarray: Масив обробленого зображення.
    """
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def apply_filters(img, model, confidence_threshold):
    """Застосування фільтрів та перевірка впевненості."""
    filters = [
        ("Контраст", ImageEnhance.Contrast(img).enhance(4)),
        ("Різкість", ImageEnhance.Sharpness(img).enhance(4)),
        ("Колір", ImageEnhance.Color(img).enhance(4)),
        ("Деталізація", img.filter(ImageFilter.DETAIL)),
        ("Додаткова різкість", img.filter(ImageFilter.SHARPEN)),
        ("Згладжування", img.filter(ImageFilter.SMOOTH)),
        ("Розмиття", img.filter(ImageFilter.GaussianBlur(1))),
        ("Накладання фонового кольору", img.convert('L').point(lambda x: x // 16 * 16))
    ]

    for filter_name, filtered_img in filters:
        img_array = preprocess_image(filtered_img)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        if confidence >= confidence_threshold:
            return f"Точність прогнозу була нижче заданого порогу, тому застосували фільтр '{filter_name}'. Результат : {os.getenv('MODEL_CLASSES', '').split(',')[predicted_class]} із вірогідністю {confidence * 100:.2f}%"

    return f"Поточне зображення не підходить для класифікації. Впевненість моделі становить: {confidence * 100:.2f}%. Завантажте, будь ласка, інше зображення."

def make_prediction(model, img_array, confidence_threshold):
    """Прогнозування на основі моделі та повернення результату."""
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    class_labels = "літак,автомобіль,птах,кішка,олень,собака,жаба,кінь,корабель,вантажівка".split(",")

    if confidence >= confidence_threshold:
        result_text = f"Результат : {class_labels[predicted_class]} із вірогідністю у {confidence * 100:.2f}%"
    else:
        result_text = f"Поточне зображення не підходить для класифікації. Впевненість моделі становить: {confidence * 100:.2f}%. Завантажте, будь ласка, інше зображення."

    return result_text, confidence
