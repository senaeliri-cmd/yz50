GPT-2 Optimization Results — NVIDIA Tesla T4
B = 8, T = 1024 · 16.29M parametre
Optimizasyon	      Ortalama süre     Token/sn	FP32'ye göre
FP32 Baseline	        343.90 ms	     23.821	    1.00×
TF32	Desteklenmiyor	                    —	        —
FP16 + GradScaler	    170.11 ms	    48.158	    2.02×
FP16 + torch.compile	96.41 ms	    84.967	    3.57×
FP16 + Compile + SDPA	80.92 ms	    101.240	    4.25×
Vocab Padding (50.304)	73.97 ms	    110.752	    4.65×



En sonda 64 minibatch e çevirme sebebimiz tek seferde 52428 token içeren bir batch in memory e sığmayacağı için hata vermesiydi. Biz de bunun yerine 52428 tokenin B ve T değerleriyle kaç mini batch te temsil edilebileceğini bulup forward backward yaparken içeride o kadar mini batch oluşturup forward ve backward yapıyoruz. Ama sonucunda weightleri sadece bir kere güncelliyoruz.