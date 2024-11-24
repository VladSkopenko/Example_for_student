import streamlit as st
from PIL import Image
import numpy as np
from load_model import model_conv
from utils import make_prediction
from utils import preprocess_image

#st.title("Візуалізація роботи нейронної мережі")

#uploaded_image = st.file_uploader("Завантажте зображення", type=["jpg", "png"])

#model_choice = st.selectbox("Оберіть модель", ["Згорткова модель", "VGG16"])

# if st.button("Класифікувати"):
#     if uploaded_image:
#
#         image = Image.open(uploaded_image)
#         st.image(image, caption="Вхідне зображення")
#
#         img_array = preprocess_image(image)
#
#
#         try:
#             model = model_conv
#         except ValueError as e:
#             st.error(str(e))
#             model = None
#
#         if model:
#             predictions = model.predict(img_array)
#             predicted_class = np.argmax(predictions)
#             st.write(f"Передбачений клас: {predicted_class}")  # Тут ще руками потрібно буде обробити , щоб не просто класс 9 а наприклад - автомобіль
#             st.write(f"Ймовірності: {predictions}")
#
#     else:
#         st.warning("Будь ласка, завантажте зображення.")

def main():
    st.title("Візуалізація роботи нейронної мережі")

    uploaded_image = st.file_uploader("Завантажте зображення", type=["jpg", "png"])

    model_choice = st.selectbox("Оберіть модель", ["Згорткова модель", "VGG16"])
    confidence_threshold = st.slider("Оберіть поріг впевненості", min_value=0.0, max_value=1.0, value=0.70, step=0.01)

    if st.button("Класифікувати"):
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Вхідне зображення")

            img_array = preprocess_image(image)

            try:
                model = model_conv
            except ValueError as e:
                st.error(str(e))
                model = None

            if model:
                result_text, confidence = make_prediction(model, img_array, confidence_threshold)
                st.write(result_text)
                st.write(f"Ймовірності: {model.predict(img_array)}")
        else:
            st.warning("Будь ласка, завантажте зображення.")


main()