import streamlit as st
from PIL import Image
from load_model import model_conv
from utils import make_prediction
from utils import preprocess_image


def main():
    st.title("Візуалізація роботи нейронної мережі")

    uploaded_image = st.file_uploader("Завантажте зображення", type=["jpg", "png"])

    model_choice = st.selectbox("Оберіть модель", ["Згорткова модель", "VGG16", "Умовна модель №3"])# Ось це впливає на кнопку з вибором моделі
    confidence_threshold = st.slider("Оберіть поріг впевненості", min_value=0.0, max_value=1.0, value=0.70, step=0.01)

    if st.button("Класифікувати"):
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Вхідне зображення")

            img_array = preprocess_image(image)

            try:
                model = model_conv # Ось тут ви можете розширити кількість моделей, додати функцію можливо яка буде обирати модель
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