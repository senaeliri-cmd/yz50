Unicode kod noktaları her karşılaştığı karakter için sayı atar. Bunun yanında UTF-8 1-2-3-4 byte olarak temsil edebiliyor karakterleri. UTF-8 böyle temsil edebildiği için aslında 256 temel token onun için yeterli oluyor. Çünkü zaten farklı byte lar sayesinde 2 ** 256 olasılık oluşturuyor. Mesela türkçe çağlar ismi için 

[199, 97, 287, 108, 97, 114] -----> Unicode temsili
[195, 135, 97, 196, 159, 108, 97, 114] -----> UTF-8 temsili

Ç ve ğ gibi harfler 2 byte ile gösterildiği için 2 farklı sayı ile UTF-8'de temsil edilirken Unicode da tek sayı ile temsil ediliyor. Biz vocab_size için embedding table oluşturduğumuzdan dolayı vocab_size i çok büyük bir tablonun embedding' i de çok büyük olur. Unicode doğrudan kullanıldığında geniş bir vocabulary gerektirebilir. UTF-8 sayesinde 256 temel byte ile her türlü Unicode metnini temsil edebiliriz. BPE ise bu byte'ları sık tekrar eden gruplar hâlinde birleştirerek token sayısını azaltır ve vocabulary büyüklüğünü kontrol etmemizi sağlar.


Sıkıştırma oranı ilk baştaki metinle sıkışmış metin arasında yaklaşık 1.3 kadardı

İlk 20 token:
Tokens (196, 177) merged into 40737
Tokens (110, 32) merged into 17033
Tokens (195, 188) merged into 16370
Tokens (97, 114) merged into 15096
Tokens (101, 32) merged into 13352
Tokens (197, 159) merged into 13347
Tokens (46, 32) merged into 13201
Tokens (101, 114) merged into 12212
Tokens (97, 32) merged into 11722
Tokens (44, 32) merged into 11621
Tokens (105, 32) merged into 10824
Tokens (105, 114) merged into 9573
Tokens (195, 167) merged into 8395
Tokens (196, 159) merged into 7807
Tokens (105, 110) merged into 7778
Tokens (97, 107) merged into 7374
Tokens (97, 110) merged into 7129
Tokens (257, 32) merged into 6551
Tokens (97, 108) merged into 6503
Tokens (257, 110) merged into 6218

1.2917909282066686
1075093
832250

"Hello've world123 how's are you!!!?"
['Hello', "'ve", ' world', '123', ' how', "'s", ' are', ' you', '!!!?']

"Kemal'den aldığı bilgiyle Hasan'ın yanına gitiğinde Ahmet'i gördü ?!"
['Kemal', "'d", 'en', ' aldığı', ' bilgiyle', ' Hasan', "'", 'ın', ' yanına', ' gitiğinde', ' Ahmet', "'", 'i', ' gördü', ' ?!']

Bu regex Türkçede yanlış parçalamalar yapıyor. Çünkü 's 't 're gibi şeyleri direkt yazarak belirtmiş ama bu eklerin Türkçe'de anlamı yok. Türkçe'de -de/-dan/-i/-a gibi ekler var.Öyle olunda mesela Ahmet'den dediğimde Ahmet alıyor 'd yapıyor en alıyor yani yanlış parçalıyor. Bunun yerine ' işaretinden sonra gelen her şey demek daha genel ve doğru olabilir. Bunun da sıkıntısı tırnak işaretinin alıntılarda da kullanılabilmesi olur. Bence tırnak işaretini komple ayrı tutup sonrasında gelen kısmın ek mi yoksa alıntı cümle mi olduğunu modele bırakmak da mantıklı bir seçenek olabilir. Bunun da kötü tarafı fazladan token kullandığı için genel olarak daha kötü sıkıştırılmış bir model olması olabilir. Ama kendi içinde yaklaşımsal olarak daha tutarlı olur. 


############## ÖRNEK METİN ############################
Ahırın avlusunda oynarken aşağıda, gümüş söğütler altında görünmeyen derenin hazin şırıltısını işitirdik. Evimiz iç çitin büyük kestane ağaçları arkasında kaybolmuş gibiydi. Annem, İstanbul'a gittiği için benden bir yaş küçük olan kardeşim Hasan'la artık Dadaruh'un yanından hiç ayrılmıyorduk. Bu, babamın seyisi, ihtiyar bir adamdı. Sabahleyin erkenden ahıra koşuyorduk. En sevdiğimiz şey atlardı. Dadaruh'la beraber onları suya götürmek, çıplak sırtlarına binmek, ne doyulmaz bir zevkti. Hasan korkar, yalnız binemezdi. Dadaruh onu kendi önüne alırdı. Torbalara arpa koymak, yemliklere ot doldurmak, gübreleri kaldırmak eğlenceli bir oyundan ziyade, bizim hoşumuza gidiyordu. Hele tımar... Bu en zevkli şeydi. Dadaruh eline kaşağıyı alıp işe başladı mı, tıpkı... tık... tık... tık... tıpkı bir saat gibi... yerimde duramaz
#######################################################

------> Türkçe 352 token. Hala "İstanbul'dan" örneğini 'da ve n olarak iki token a ayırıyor. Direkt 'dan diye algılayamıyor


Aynı örnek metnin ingilizcesindeyse 229 tokena ayırabiliyor.

Benim videoda görüp kullandığım gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") regexi ile 146 token a ayrıldı.
Bunun ana sebebi de kelimeleri hiç bölmeden direkt kullanmasıydı.
