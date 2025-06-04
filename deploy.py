import pickle
import streamlit as st

# Load model TF-IDF
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load model SVM
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Judul Aplikasi
st.title("Analisis Sentimen ⚽︎")
st.write("Masukkan kalimat di bawah ini, dan sistem akan mendeteksi apakah kalimat tersebut positif atau negatif.")

# Input teks dari pengguna
text_input = st.text_area("✍️Masukkan kalimat Anda:", height=150)

if st.button("Prediksi Sentimen"):
    if text_input:
        # Transformasi teks baru ke TF-IDF
        text_tfidf = vectorizer.transform([text_input])

        # Prediksi dengan model SVM
        prediction = svm_model.predict(text_tfidf)[0]

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if prediction == 1:
            st.success("😊 Sentimen Positif")
        elif prediction == -1:
            st.error("😠 Sentimen Negatif")
        else:
            st.warning("Sentimen tidak teridentifikasi.")
    else:
        st.error("Silakan masukkan teks terlebih dahulu.")

# Tambahkan footer atau informasi tambahan
st.markdown("---")
st.write("### Tentang Aplikasi")
st.write("Aplikasi ini menggunakan model SVM untuk menganalisis sentimen dari kalimat yang Anda masukkan. "
         "Silakan coba kalimat yang berbeda untuk melihat hasilnya!")
st.write("### Pengembang")
st.write("Dikembangkan oleh: Isma Magfirotul Yuna 21.12.1871")