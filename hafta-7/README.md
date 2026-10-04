Görev1:
Base val loss = 2.6305

Görev 2:
Biz bir sonraki karakteri hesaplarken kendinen önceki tüm karakterlere bakabilmek istiyoruz. şuan mevcut karakter sayımızda bir matrix oluşturuyoruz (T,T) sonrasında bu karakterlerden bakmasına izin vermeyi 1 olarak düşünelim bakma demeyi 0 ben 8 karakterden ilkinin sadece kendine bakmasını istiyorum diğerinin kendi ve 1ciye derken üçgen şeklinde bir izin tablosu oluşuyor. Bunu da a = torch.tril(torch.ones(T,T)) ile oluşturuyorum. Bir sonraki soru ise tamam ben nelere bakacağını buldum ama ne ağırlıkta bakacak? Eğer 3 elemente bakıyorsa bunların ortalamasını alması lazım olduğu için a = a / a.sum(1, keepdim=True) burada amacım tamam ben kendimden önceki 4 karakterin channellarına bakacağım ve bu channel lardan aldığım ortalamayı yeni karakterin channelı yapacağım. şimdi elimde ağırlıklı izin tablosu a var ve channel değerlerinin tutulduğu x ben bu weightlerle ağırlıklı tabloyu çarparsam istediğim karakterin kendinden önceki karakterlere bakarak elde ettiği yeni channel değerlerine ulaşırım.

Görev 3:
| 6. tokenın baktığı token | Scaling yok | ÷ √16 (÷4) | Değişim |
|---|---:|---:|---:|
| Token 1 | **40.86%** | **21.89%** | ↓ 18.97 puan |
| Token 2 | 5.16% | 13.05% | ↑ 7.89 puan |
| Token 3 | 8.33% | 14.71% | ↑ 6.38 puan |
| Token 4 | 9.91% | 15.36% | ↑ 5.45 puan |
| Token 5 | **25.12%** | **19.38%** | ↓ 5.74 puan |
| Token 6 (kendisi) | 10.61% | 15.62% | ↑ 5.01 puan |
| Token 7 (gelecek) | 0% | 0% | Maskeli |
| Token 8 (gelecek) | 0% | 0% | Maskeli |

Tabloda gördüğümüz gibi scale yapmadan önce 0.40 attention ağırlığı verdiği token, scale işleminden sonra 0.21'e düşüyor. Buradaki amacımız, bazı attention skorlarının diğerlerine kıyasla çok yüksek değerlere ulaşıp softmax sonucunun aşırı keskinleşmesini önlemek. Skorları ölçeklendirerek attention ağırlıklarının daha dengeli dağılmasını sağlıyoruz.


Görev 4:
Sadece kendinden önceki karaktere bakan val loss = val 2.6305'ti. Görev 4'te yazdığımız attention mekanizmasına sahip val loss = 2.4265

Val loss da düşüş yaşanmasının sebebi yeni modelde karakter tahmini yaparken single batch'teki 8 karakterin her biri kendisi ve kendisinden öne gelen karakterleri bakarak tahmin yapmaya başladılar.

Generate:
Yeni karakter ürettikçe hepsini idx'e ekliyorum. Ama modelin context uzunluğu block_size ile sınırlı ve attention maskesi de buna göre oluşturuluyor. Bu yüzden modele her seferinde idx'in yalnızca son block_size tokenını veriyorum.