"""
Simulasi Komputasi Naïve Bayes Kategorikal & Evaluasi Kinerja Model (Play Tennis).

Fitur Komputasi:
1. Penurunan frekuensi & probabilitas prior kelas.
2. Estimasi likelihood empiris MLE vs Laplace Smoothing (alpha = 1).
3. Demonstrasi krisis probabilitas nol (Zero-Frequency Problem) pada kondisi Cuaca = Mendung.
4. Evaluasi performa klasifikasi (Confusion Matrix & Metrik Kinerja) langsung pada dataset Play Tennis.
"""

from typing import Dict, List, Tuple


def get_play_tennis_dataset() -> List[Dict[str, str]]:
    """Mengembalikan dataset kanonik Play Tennis (Tom Mitchell, 1997)."""
    return [
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


CARDINALITY: Dict[str, int] = {
    "Cuaca": 3,       # Cerah, Mendung, Hujan
    "Suhu": 3,        # Panas, Sedang, Sejuk
    "Kelembaban": 2,  # Tinggi, Normal
    "Angin": 2,       # Lemah, Kuat
}


def compute_priors(dataset: List[Dict[str, str]]) -> Tuple[float, float, int, int, int]:
    total_data = len(dataset)
    count_ya = sum(1 for d in dataset if d["Main"] == "Ya")
    count_tidak = sum(1 for d in dataset if d["Main"] == "Tidak")
    prior_ya = count_ya / total_data
    prior_tidak = count_tidak / total_data
    return prior_ya, prior_tidak, count_ya, count_tidak, total_data


def predict_sample(
    sample: Dict[str, str],
    dataset: List[Dict[str, str]],
    alpha: float = 0.0,
) -> Dict[str, float]:
    """
    Melakukan inferensi Naïve Bayes kategorikal dengan atau tanpa Laplace Smoothing.
    """
    prior_ya, prior_tidak, count_ya, count_tidak, _ = compute_priors(dataset)

    lik_ya = 1.0
    lik_tidak = 1.0
    feature_details: Dict[str, Dict[str, float]] = {}

    for feat in ["Cuaca", "Suhu", "Kelembaban", "Angin"]:
        val = sample[feat]
        c_ya = sum(1 for d in dataset if d["Main"] == "Ya" and d[feat] == val)
        c_tidak = sum(1 for d in dataset if d["Main"] == "Tidak" and d[feat] == val)
        card = CARDINALITY[feat]

        if alpha == 0.0:
            p_ya = c_ya / count_ya
            p_tidak = c_tidak / count_tidak
        else:
            p_ya = (c_ya + alpha) / (count_ya + alpha * card)
            p_tidak = (c_tidak + alpha) / (count_tidak + alpha * card)

        lik_ya *= p_ya
        lik_tidak *= p_tidak

        feature_details[feat] = {
            "p_ya": p_ya,
            "p_tidak": p_tidak,
            "count_ya": c_ya,
            "count_tidak": c_tidak,
        }

    post_ya = prior_ya * lik_ya
    post_tidak = prior_tidak * lik_tidak
    total_post = post_ya + post_tidak

    prob_ya = post_ya / total_post if total_post > 0 else 0.5
    prob_tidak = post_tidak / total_post if total_post > 0 else 0.5

    return {
        "lik_ya": lik_ya,
        "lik_tidak": lik_tidak,
        "post_ya": post_ya,
        "post_tidak": post_tidak,
        "prob_ya": prob_ya,
        "prob_tidak": prob_tidak,
        "prediction": "Ya" if post_ya >= post_tidak else "Tidak",
        "feature_details": feature_details,
    }


def run_discrete_nb_simulation() -> None:
    print("=" * 75)
    print(" SIMULASI STEP-BY-STEP NAÏVE BAYES: DATASET KANONIK 'PLAY TENNIS'")
    print("=" * 75)

    dataset = get_play_tennis_dataset()
    prior_ya, prior_tidak, count_ya, count_tidak, total_data = compute_priors(dataset)

    print(f"\n1. DISTRIBUSI PRIOR KELAS (N = {total_data}):")
    print(f"   P(Main = Ya)    = {count_ya}/{total_data} = {prior_ya:.4f} (64.29%)")
    print(f"   P(Main = Tidak) = {count_tidak}/{total_data} = {prior_tidak:.4f} (35.71%)")

    # Instance Uji 1: Standar Mitchell (1997)
    sample_1 = {"Cuaca": "Cerah", "Suhu": "Sejuk", "Kelembaban": "Tinggi", "Angin": "Kuat"}
    print(f"\n2. KASUS 1 (STANDAR TANPA FREKUENSI NOL): {sample_1}")
    res_1_raw = predict_sample(sample_1, dataset, alpha=0.0)
    res_1_laplace = predict_sample(sample_1, dataset, alpha=1.0)

    print("   [A] Estimasi MLE (Tanpa Smoothing):")
    print(f"       Likelihood : P(x|Ya) = {res_1_raw['lik_ya']:.6f}, P(x|Tidak) = {res_1_raw['lik_tidak']:.6f}")
    print(f"       Posterior  : P(Ya|x) = {res_1_raw['prob_ya']*100:.2f}%, P(Tidak|x) = {res_1_raw['prob_tidak']*100:.2f}%")
    print(f"       Prediksi   : {res_1_raw['prediction']}")

    print("   [B] Estimasi Laplace Smoothing (alpha = 1):")
    print(f"       Likelihood : P(x|Ya) = {res_1_laplace['lik_ya']:.6f}, P(x|Tidak) = {res_1_laplace['lik_tidak']:.6f}")
    print(f"       Posterior  : P(Ya|x) = {res_1_laplace['prob_ya']*100:.2f}%, P(Tidak|x) = {res_1_laplace['prob_tidak']*100:.2f}%")
    print(f"       Prediksi   : {res_1_laplace['prediction']}")

    # Instance Uji 2: Demonstrasi Nyata Zero-Frequency Problem (Cuaca = Mendung)
    sample_zero = {"Cuaca": "Mendung", "Suhu": "Sejuk", "Kelembaban": "Tinggi", "Angin": "Kuat"}
    print(f"\n3. KASUS 2 (DEMONSTRASI ZERO-FREQUENCY PROBLEM DENGAN Cuaca = Mendung): {sample_zero}")
    res_z_raw = predict_sample(sample_zero, dataset, alpha=0.0)
    res_z_laplace = predict_sample(sample_zero, dataset, alpha=1.0)

    print("   [A] Estimasi MLE Mentah (Tanpa Smoothing):")
    print("       Frekuensi Cuaca=Mendung pada data latih: Ya=4/9, Tidak=0/5 (NOL MUTLAK!)")
    print(f"       Likelihood Kelas Tidak: P(Mendung|Tidak) = 0/5 = 0.0000 -> P(x|Tidak) = {res_z_raw['lik_tidak']:.6f}")
    print(f"       Posterior Relatif : psi(Ya) = {res_z_raw['post_ya']:.6f}, psi(Tidak) = {res_z_raw['post_tidak']:.6f}")
    print(f"       Probabilitas Posterior : P(Ya|x) = {res_z_raw['prob_ya']*100:.2f}%, P(Tidak|x) = {res_z_raw['prob_tidak']*100:.2f}%")
    print("       ANALISIS: Terjadi keruntuhan peluang (probability collapse)! Karena satu nilai nol,")
    print("                 seluruh bukti fitur Kelembaban=Tinggi dan Angin=Kuat terhapus.")

    print("\n   [B] Solusi Laplace Smoothing (alpha = 1):")
    print("       P_Laplace(Mendung|Tidak) = (0 + 1) / (5 + 3) = 1/8 = 0.1250 > 0")
    print(f"       Likelihood Kelas Tidak: P_Laplace(x|Tidak) = {res_z_laplace['lik_tidak']:.6f} > 0")
    print(f"       Posterior Relatif : psi(Ya) = {res_z_laplace['post_ya']:.6f}, psi(Tidak) = {res_z_laplace['post_tidak']:.6f}")
    print(f"       Probabilitas Posterior : P(Ya|x) = {res_z_laplace['prob_ya']*100:.2f}%, P(Tidak|x) = {res_z_laplace['prob_tidak']*100:.2f}%")
    print("       ANALISIS: Zero-Frequency Problem teratasi secara tuntas tanpa menghilangkan bobot bukti.")

    # 4. Evaluasi Kinerja pada Dataset Play Tennis
    print("\n4. EVALUASI PERFORMA MODEL PADA DATASET PLAY TENNIS (N = 14):")
    tp, tn, fp, fn = 0, 0, 0, 0
    for d in dataset:
        res = predict_sample(d, dataset, alpha=1.0)
        pred = res["prediction"]
        act = d["Main"]
        if act == "Ya" and pred == "Ya":
            tp += 1
        elif act == "Tidak" and pred == "Tidak":
            tn += 1
        elif act == "Tidak" and pred == "Ya":
            fp += 1
        elif act == "Ya" and pred == "Tidak":
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print("   Matriks Konfusi (Play Tennis):")
    print(f"   - True Positive  (TP) : {tp}")
    print(f"   - True Negative  (TN) : {tn}")
    print(f"   - False Positive (FP) : {fp}")
    print(f"   - False Negative (FN) : {fn}")
    print(f"   Akurasi   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Presisi   : {prec:.4f} ({prec*100:.2f}%)")
    print(f"   Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"   F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    run_discrete_nb_simulation()
