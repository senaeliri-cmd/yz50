# Week 6 - WaveNet

Bu hafta önceki MLP modelini WaveNet benzeri hiyerarşik bir yapıya dönüştürdüm. Context uzunluğunu 3'ten 8'e çıkardım, `FlattenConsecutive` ile karakterleri kademeli olarak birleştirdim ve `BatchNorm1d`'nin 3D tensorlar için çalışma şeklini düzelttim.

## Model Results

| Model | Train Loss | Dev Loss | Parameters |
|---|---:|---:|---:|
| Week 4 - 3-character MLP | 2.021 | 2.329 | - |
| 3-character MLP | 2.018 | 2.323 | 12,097 |
| 8-character MLP | 1.878 | 2.245 | 22,097 |
| 8-character WaveNet | 1.882 | 2.251 | - |
| 8-character WaveNet - Fixed BatchNorm1d | 1.875 | 2.238 | 22,397 |
| 8-character WaveNet - 24 embedding, 128 hidden | 1.743 | 2.226 | - |

## WaveNet Architecture

8-character context için katmanların shape değişimleri:

| Layer | Output Shape |
|---|---|
| Embedding | `(32, 8, 10)` |
| FlattenConsecutive | `(32, 4, 20)` |
| Linear | `(32, 4, 68)` |
| BatchNorm1d | `(32, 4, 68)` |
| Tanh | `(32, 4, 68)` |
| FlattenConsecutive | `(32, 2, 136)` |
| Linear | `(32, 2, 68)` |
| BatchNorm1d | `(32, 2, 68)` |
| Tanh | `(32, 2, 68)` |
| FlattenConsecutive | `(32, 136)` |
| Linear | `(32, 68)` |
| BatchNorm1d | `(32, 68)` |
| Tanh | `(32, 68)` |
| Linear | `(32, 27)` |

Amacımız tüm önceki karakterlerden gelen bilgileri tek seferde sıkıştırmak yerine, ikişer ardışık karakterin bilgisini kademeli olarak birleştirerek ilerlemek.

Bu şekilde receptive field her katmanda büyüyor:

`2 → 4 → 8`

Böylece model daha uzak geçmişteki karakterlerden gelen bilgileri hiyerarşik olarak kullanabiliyor.

## BatchNorm1d Fix

`BatchNorm1d` ilk yazıldığında girdisi `Flatten` ile 2D bir tensöre dönüştürülüyordu. `FlattenConsecutive` kullanmaya başladıktan sonra ise BatchNorm'a `(batch, context, feature)` şeklinde 3D tensorlar gelmeye başladı.

Örneğin ilk BatchNorm'a:

`(32, 4, 68)`

boyutunda bir tensor geliyor:

- `32`: batch size
- `4`: context pozisyonu
- `68`: neuron / feature sayısı

BatchNorm her feature için batch ve context boyunca ortalama almalıdır. Dolayısıyla her feature için:

`32 × 4 = 128`

değerden tek bir mean ve variance hesaplanmalıdır.

Sadece `dim=0` üzerinden ortalama aldığımızda context dimensionı korunuyordu. Bu nedenle her feature için tek bir ortalama yerine 4 farklı ortalama hesaplanıyordu.

Bunu düzeltmek için:

- 2D input → `dim=0`
- 3D input → `dim=(0, 1)`

kullanıldı.

### Result

| Version | Dev Loss |
|---|---:|
| Before BatchNorm1d fix | 2.251 |
| After BatchNorm1d fix | 2.238 |

BatchNorm düzeltmesinden sonra dev loss `2.251 → 2.238` oldu.

## Model Comparison

Görev kapsamında üç temel mimarinin karşılaştırması:

| Model | Parameters | Dev Loss |
|---|---:|---:|
| 3-context MLP | 12,097 | 2.323 |
| 8-context MLP | 22,097 | 2.245 |
| 8-context WaveNet | 22,397 | 2.238 |

Context uzunluğunu 3'ten 8'e çıkarmak dev loss'u düşürdü. WaveNet ise 8-context düz MLP'ye yakın parametre sayısıyla biraz daha düşük dev loss elde etti.

## Turkish Name Generation

Türkçe isim dataseti üzerinde aynı WaveNet mimarisi eğitildi.

| Model | Dev Loss |
|---|---:|
| Week 4 - 3-context Turkish MLP | 2.329 |
| Week 6 - 8-context Turkish WaveNet | 2.312 |

### Generated Names

- erminer
- gülbey
- akköl
- gülser
- alpay
- cavsol
- demirdoğa
- nazlı
- pesuzan
- uçe

Türkçe modelde context uzunluğunu 3'ten 8'e çıkarıp WaveNet yapısını kullandığımızda dev loss `2.329`'dan `2.312`'ye düştü.

Bu yaklaşık `0.017`'lik küçük bir iyileşme sağladı. Daha uzun context sayesinde model, bir sonraki karakteri tahmin ederken yalnızca son 3 karakter yerine son 8 karaktere kadar olan geçmiş bilgiyi kullanabildi. Böylece daha uzun karakter örüntülerini ve bağımlılıklarını modelleme imkânı kazandı.