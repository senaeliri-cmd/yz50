Unicode kod noktaları her karşılaştığı karakter için sayı atar. Bunun yanında UTF-8 1-2-3-4 byte olarak temsil edebiliyor karakterleri. UTF-8 böyle temsil edebildiği için aslında 256 temel token onun için yeterli oluyor. Çünkü zaten farklı byte lar sayesinde 2 ** 256 olasılık oluşturuyor. Mesela türkçe çağlar ismi için 

[199, 97, 287, 108, 97, 114] -----> Unicode temsili
[195, 135, 97, 196, 159, 108, 97, 114] -----> UTF-8 temsili

Ç ve ğ gibi harfler 2 byte ile gösterildiği için 2 farklı sayı ile UTF-8'de temsil edilirken Unicode da tek sayı ile temsil ediliyor. Biz vocab_size için embedding table oluşturduğumuzdan dolayı vocab_size i çok büyük bir tablonun embedding' i de çok büyük olur. Bu da çok fazla yer kaplayacağı ve verimsiz olacağı için daha az yer tutan ve ne kadar yer kaplayacağını bildiğimiz-ne kadar sıkıştıracağımı kendim seçeceğim için- şekilde BPE'de temsil etmek daha mantıklı olur.