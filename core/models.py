from django.db import models

class RiwayatAnalisis(models.Model):
    jam_belajar = models.FloatField()
    jam_tidur = models.FloatField()
    tugas_mingguan = models.FloatField()
    hasil_ml = models.CharField(max_length=50)
    skor_fuzzy = models.FloatField()
    penjelasan_ai = models.TextField()
    dibuat_pada = models.DateTimeField(auto_now_add=True)