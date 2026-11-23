import os, io, base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq
from django.conf import settings
from django.shortcuts import render
from .models import RiwayatAnalisis
from .ml_fuzzy import jalankan_ml_dan_fuzzy

load_dotenv()
DATASET_PATH = os.path.join(settings.BASE_DIR, "dataset_proyek.csv")

def buat_grafik_dan_ringkasan(df, input_user=None):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    rata_rata = df[["jam_belajar", "jam_tidur", "tugas_mingguan"]].mean()

    if input_user:
        perbandingan = pd.DataFrame({
            "Rata-rata Dataset": rata_rata.values,
            "Input Pengguna": input_user
        }, index=["Jam Belajar", "Jam Tidur", "Jumlah Tugas"])
        perbandingan.plot(kind="bar", ax=ax, color=["#4a90e2", "#e67e22"])
        ringkasan = (
            f"Data grafik membandingkan Input Pengguna (Belajar={input_user[0]} jam, "
            f"Tidur={input_user[1]} jam, Tugas={input_user[2]}) dengan Rata-rata Dataset "
            f"(Belajar={rata_rata['jam_belajar']:.1f}, Tidur={rata_rata['jam_tidur']:.1f}, "
            f"Tugas={rata_rata['tugas_mingguan']:.1f})."
        )
    else:
        rata_rata.plot(kind="bar", ax=ax, color="#4a90e2")
        ringkasan = (
            f"Grafik menampilkan rata-rata keseluruhan dataset: Jam Belajar={rata_rata['jam_belajar']:.1f}, "
            f"Jam Tidur={rata_rata['jam_tidur']:.1f}, Tugas Mingguan={rata_rata['tugas_mingguan']:.1f}."
        )

    ax.set_title("Perbandingan Parameter Analisis")
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64, ringkasan

def jelaskan_dengan_ai(ringkasan_grafik, hasil_ml="", skor_fuzzy=""):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "API Key Groq belum diatur di file .env."
    try:
        client = Groq(api_key=api_key)
        daftar_model = [
            m.id for m in client.models.list().data
            if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "playai", "orpheus"])
        ]
        if not daftar_model:
            return "Tidak ada model aktif di akun Groq."

        prioritas = [m for m in daftar_model if any(k in m.lower() for k in ["llama", "qwen", "gpt-oss", "gemma"])]
        model_aktif = prioritas[0] if prioritas else daftar_model[0]

        prompt = (
            f"{ringkasan_grafik} Hasil ML: {hasil_ml}, Skor Fuzzy: {skor_fuzzy}/100. "
            "Jelaskan makna grafik perbandingan tersebut dan berikan saran singkat "
            "dalam 1 paragraf bahasa Indonesia yang jelas."
        )
        res = client.chat.completions.create(
            model=model_aktif,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Gagal memuat penjelasan AI: {e}"

def index(request):
    df = pd.read_csv(DATASET_PATH)
    hasil_ml = None
    skor_fuzzy = None

    if request.method == "POST":
        jb = float(request.POST.get("jam_belajar"))
        jt = float(request.POST.get("jam_tidur"))
        tm = float(request.POST.get("tugas_mingguan"))

        hasil_ml, skor_fuzzy = jalankan_ml_dan_fuzzy(jb, jt, tm)
        grafik, ringkasan = buat_grafik_dan_ringkasan(df, [jb, jt, tm])
        penjelasan_ai = jelaskan_dengan_ai(ringkasan, hasil_ml, skor_fuzzy)

        RiwayatAnalisis.objects.create(
            jam_belajar=jb, jam_tidur=jt, tugas_mingguan=tm,
            hasil_ml=hasil_ml, skor_fuzzy=skor_fuzzy, penjelasan_ai=penjelasan_ai
        )
    else:
        grafik, ringkasan = buat_grafik_dan_ringkasan(df)
        penjelasan_ai = "Silakan masukkan data pada form di samping untuk melihat penjelasan AI terhadap grafik."

    riwayat = RiwayatAnalisis.objects.all().order_by("-dibuat_pada")[:5]
    return render(request, "index.html", {
        "grafik": grafik,
        "hasil_ml": hasil_ml,
        "skor_fuzzy": skor_fuzzy,
        "penjelasan_ai": penjelasan_ai,
        "riwayat": riwayat
    })