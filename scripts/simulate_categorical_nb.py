"""
Simulasi Komputasi Naïve Bayes Kategorikal & Evaluasi Kinerja Model (Play Tennis).

Mata Kuliah: Kecerdasan Buatan (Tugas 3: Project-Based Learning)
Kelompok 3 - S-1 Matematika FMIPA UNS

Fitur:
1. Perhitungan manual step-by-step Naïve Bayes kategorikal.
2. Penerapan Laplace Smoothing (penanganan Zero Frequency Problem).
3. Perhitungan Confusion Matrix & Metrik Evaluasi (Accuracy, Precision, Recall, F1-Score).
"""

from typing import Dict, List


def run_discrete_nb_simulation() -> None:
    print("=" * 65)
    print(" SIMULASI STEP-BY-STEP NAÏVE BAYES: DATASET DUMMY 'PLAY TENNIS'")
    print("=" * 65)

    dataset: List[Dict[str, str]] = [
        {"Cuaca": "Cerah", "Suhu": "Panas", "Kelembaban": "Tinggi", "Angin": "Lemah", "Main": "Tidak"},
        {"Cuaca": "Cerah", "Suhu": "Panas", "Kelembaban": "Tinggi", "Angin": "Kuat", "Main": "Tidak"},
        {"Cuaca": "Mendung", "Suhu": "Panas", "Kelembaban": "Tinggi", "Angin": "Lemah", "Main": "Ya"},
        {"Cuaca": "Hujan", "Suhu": "Sedang", "Kelembaban": "Tinggi", "Angin": "Lemah", "Main": "Ya"},
        {"Cuaca": "Hujan", "Suhu": "Sejuk", "Kelembaban": "Normal", "Angin": "Lemah", "Main": "Ya"},
        {"Cuaca": "Hujan", "Suhu": "Sejuk", "Kelembaban": "Normal", "Angin": "Kuat", "Main": "Tidak"},
        {"Cuaca": "Mendung", "Suhu": "Sejuk", "Kelembaban": "Normal", "Angin": "Kuat", "Main": "Ya"},
        {"Cuaca": "Cerah", "Suhu": "Sedang", "Kelembaban": "Tinggi", "Angin": "Lemah", "Main": "Tidak"},
        {"Cuaca": "Cerah", "Suhu": "Sejuk", "Kelembaban": "Normal", "Angin": "Lemah", "Main": "Ya"},
        {"Cuaca": "Hujan", "Suhu": "Sedang", "Kelembaban": "Normal", "Angin": "Lemah", "Main": "Ya"},
        {"Cuaca": "Cerah", "Suhu": "Sedang", "Kelembaban": "Normal", "Angin": "Kuat", "Main": "Ya"},
        {"Cuaca": "Mendung", "Suhu": "Sedang", "Kelembaban": "Tinggi", "Angin": "Kuat", "Main": "Ya"},
        {"Cuaca": "Mendung", "Suhu": "Panas", "Kelembaban": "Normal", "Angin": "Lemah", "Main": "Ya"},
        {"Cuaca": "Hujan", "Suhu": "Sedang", "Kelembaban": "Tinggi", "Angin": "Kuat", "Main": "Tidak"},
    ]

    total_data = len(dataset)
    count_ya = sum(1 for d in dataset if d["Main"] == "Ya")
    count_tidak = sum(1 for d in dataset if d["Main"] == "Tidak")

    # 1. Prior Probability
    prior_ya = count_ya / total_data
    prior_tidak = count_tidak / total_data

    print(f"\n1. PROBABILITAS PRIOR:")
    print(f"   P(Main = Ya)    = {count_ya}/{total_data} = {prior_ya:.4f}")
    print(f"   P(Main = Tidak) = {count_tidak}/{total_data} = {prior_tidak:.4f}")

    # Instance uji baru: Cuaca = Cerah, Suhu = Sejuk, Kelembaban = Tinggi, Angin = Kuat
    uji = {"Cuaca": "Cerah", "Suhu": "Sejuk", "Kelembaban": "Tinggi", "Angin": "Kuat"}
    print(f"\n2. KONDISI DATA UJI BARU: {uji}")

    # 2. Likelihood tanpa smoothing
    print("\n3. PERHITUNGAN LIKELIHOOD:")
    likelihood_ya = 1.0
    likelihood_tidak = 1.0

    for feat, val in uji.items():
        c_feat_ya = sum(1 for d in dataset if d["Main"] == "Ya" and d[feat] == val)
        c_feat_tidak = sum(1 for d in dataset if d["Main"] == "Tidak" and d[feat] == val)

        p_ya = c_feat_ya / count_ya
        p_tidak = c_feat_tidak / count_tidak

        likelihood_ya *= p_ya
        likelihood_tidak *= p_tidak

        print(f"   P({feat} = {val} | Ya)    = {c_feat_ya}/{count_ya} = {p_ya:.4f}")
        print(f"   P({feat} = {val} | Tidak) = {c_feat_tidak}/{count_tidak} = {p_tidak:.4f}")

    # 3. Posterior Unnormalized
    post_ya = prior_ya * likelihood_ya
    post_tidak = prior_tidak * likelihood_tidak

    print("\n4. NILAI POSTERIOR RELATIF (MAP):")
    print(f"   P(Ya | X)    ~ Prior * Likelihood = {prior_ya:.4f} * {likelihood_ya:.6f} = {post_ya:.6f}")
    print(f"   P(Tidak | X) ~ Prior * Likelihood = {prior_tidak:.4f} * {likelihood_tidak:.6f} = {post_tidak:.6f}")

    prediksi = "Ya" if post_ya > post_tidak else "Tidak"
    print(f"\n>> KEPUTUSAN PREDIKSI AKHIR: {prediksi}")


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> None:
    print("\n" + "=" * 65)
    print(" EVALUASI PERFORMA MODEL (CONFUSION MATRIX)")
    print("=" * 65)
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Akurasi   : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Presisi   : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")


if __name__ == "__main__":
    run_discrete_nb_simulation()
    calculate_metrics(tp=32, tn=84, fp=16, fn=22)
