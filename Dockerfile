# Menggunakan base image Python versi 3.9.2-slim
FROM python:3.9.2-slim

# Set environment variable untuk unbuffered output dari Python
ENV PYTHONUNBUFFERED TRUE

# Menetapkan working directory di dalam container
WORKDIR /app

# Menyalin file-file yang dibutuhkan ke dalam container
COPY . .
COPY user_rating_clean.csv /app

# Menginstal paket yang diperlukan
RUN pip install --upgrade pip \
    && pip install tensorflow==2.15.0 \
    && pip install keras==2.15.0\
    && pip install gunicorn==21.2.0\
    && pip install Flask==3.0.0\
    && pip install scikit-learn==1.3.2\
    && pip install pandas==2.1.4\
    && pip install numpy==1.26.2

# Expose port yang digunakan oleh aplikasi Flask
EXPOSE 5000

# Perintah yang akan dijalankan saat container dijalankan
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]