import os
import joblib
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "model_ml.joblib")
model = joblib.load(MODEL_PATH)

def jalankan_ml_dan_fuzzy(jam_belajar, jam_tidur, tugas_mingguan):
    data_input = pd.DataFrame([{
        "jam_belajar": jam_belajar,
        "jam_tidur": jam_tidur,
        "tugas_mingguan": tugas_mingguan
    }])
    pred_kode = model.predict(data_input)[0]
    label_map = {0: "Stres Rendah", 1: "Stres Sedang", 2: "Stres Tinggi"}
    hasil_ml = label_map.get(pred_kode, "Tidak Diketahui")

    tidur = ctrl.Antecedent(np.arange(0, 13, 1), 'tidur')
    tugas = ctrl.Antecedent(np.arange(0, 21, 1), 'tugas')
    beban = ctrl.Consequent(np.arange(0, 101, 1), 'beban')

    tidur['kurang'] = fuzz.trimf(tidur.universe, [0, 0, 6])
    tidur['cukup'] = fuzz.trimf(tidur.universe, [5, 7, 9])
    tidur['banyak'] = fuzz.trimf(tidur.universe, [8, 12, 12])

    tugas['sedikit'] = fuzz.trimf(tugas.universe, [0, 0, 6])
    tugas['sedang'] = fuzz.trimf(tugas.universe, [4, 9, 14])
    tugas['banyak'] = fuzz.trimf(tugas.universe, [12, 20, 20])

    beban['ringan'] = fuzz.trimf(beban.universe, [0, 0, 45])
    beban['menengah'] = fuzz.trimf(beban.universe, [35, 55, 75])
    beban['berat'] = fuzz.trimf(beban.universe, [65, 100, 100])

    rule1 = ctrl.Rule(tidur['kurang'] & tugas['banyak'], beban['berat'])
    rule2 = ctrl.Rule(tidur['cukup'] & tugas['sedang'], beban['menengah'])
    rule3 = ctrl.Rule(tidur['banyak'] & tugas['sedikit'], beban['ringan'])
    rule4 = ctrl.Rule(tidur['kurang'] | tugas['banyak'], beban['berat'])
    rule5 = ctrl.Rule(tidur['cukup'] | tidur['banyak'], beban['ringan'])

    sistem_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    simulasi = ctrl.ControlSystemSimulation(sistem_ctrl)

    simulasi.input['tidur'] = min(max(jam_tidur, 0), 12)
    simulasi.input['tugas'] = min(max(tugas_mingguan, 0), 20)
    simulasi.compute()

    skor_fuzzy = round(float(simulasi.output['beban']), 2)
    return hasil_ml, skor_fuzzy