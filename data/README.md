# Data

Dataset yang digunakan adalah **Framingham Heart Study** yang berisi data
kesehatan 4.240 partisipan untuk memprediksi risiko penyakit jantung koroner
(CHD) dalam 10 tahun ke depan.

- File: `framingham.csv`
- Baris: 4.240
- Kolom: 16 (15 fitur + 1 target)
- Sumber: [Framingham Heart Study](https://framinghamheartstudy.org/)
- Target: `TenYearCHD` (0 = tidak mengalami CHD, 1 = mengalami CHD dalam 10 tahun)

> Dataset ini bersifat publik dan banyak digunakan untuk keperluan edukasi
> machine learning. Harap merujuk ke sumber asli untuk sitasi formal.

## Daftar Kolom

| Kolom | Deskripsi |
|-------|-----------|
| `male` | Jenis kelamin (1 = laki-laki, 0 = perempuan) |
| `age` | Usia (tahun) |
| `education` | Tingkat pendidikan (1-4) |
| `currentSmoker` | Perokok aktif saat ini (1 = ya, 0 = tidak) |
| `cigsPerDay` | Jumlah rokok per hari |
| `BPMeds` | Mengonsumsi obat tekanan darah |
| `prevalentStroke` | Riwayat stroke |
| `prevalentHyp` | Riwayat hipertensi |
| `diabetes` | Riwayat diabetes |
| `totChol` | Kolesterol total (mg/dL) |
| `sysBP` | Tekanan darah sistolik (mmHg) |
| `diaBP` | Tekanan darah diastolik (mmHg) |
| `BMI` | Indeks Massa Tubuh |
| `heartRate` | Detak jantung (bpm) |
| `glucose` | Gula darah (mg/dL) |
| `TenYearCHD` | Target: risiko CHD 10 tahun (0/1) |

## Missing Values

Beberapa kolom memiliki missing values (misalnya `glucose`, `education`,
`cigsPerDay`) yang ditangani saat preprocessing menggunakan imputasi
(median untuk numerik, modus untuk kategorikal).

## Download Ulang

Jika dataset hilang, download dari sumber publik. Proyek lokal membutuhkan
file `data/framingham.csv` sebelum menjalankan training.
