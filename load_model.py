import tensorflow as tf
from tensorflow.keras.models import load_model

model_conv = load_model("some_model.keras")

# Тту ще потрібно буде завантажити функції втрат , точність і інші параметри навчання , бо вони не зберігаються в моделі в самому файлі