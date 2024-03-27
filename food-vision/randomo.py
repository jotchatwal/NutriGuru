import streamlit as st
import tensorflow as tf

def main():
    st.title('Model Loading Test')

    # File path to the TensorFlow model
    model_file_path = "../models/EfficientNetB1.hdf5"

    # Attempt to load the model
    try:
        model = tf.keras.models.load_model(model_file_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

if __name__ == '__main__':
    main()
