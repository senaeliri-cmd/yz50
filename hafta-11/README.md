Using device: mps
Initial loss: 10.8283
Expected loss: 10.8249
Step 0 | Loss: 10.8283
Step 50 | Loss: 2.5548
Step 100 | Loss: 0.2575
Step 150 | Loss: 0.1046
Step 200 | Loss: 0.0670
Step 250 | Loss: 0.0496
Step 300 | Loss: 0.0396
Step 350 | Loss: 0.0333
Step 400 | Loss: 0.0290
Step 450 | Loss: 0.0259
Step 500 | Loss: 0.0235
Step 550 | Loss: 0.0217
Training complete!


GPT-2 Optimization Results — NVIDIA Tesla T4
B = 8, T = 1024 · 16.29M parametre
Optimizasyon	Ortalama süre	Token/sn	FP32'ye göre
FP32 Baseline	343.90 ms	23.821	1.00×
TF32	Desteklenmiyor	—	—
FP16 + GradScaler	170.11 ms	48.158	2.02×
FP16 + torch.compile	96.41 ms	84.967	3.57×
FP16 + Compile + SDPA	80.92 ms	101.240	4.25×
Vocab Padding (50.304)	73.97 ms	110.752	4.65×