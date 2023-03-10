# Association Rule Based Recommender System

"""
İş Problemi:
Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
ulaşılmasını sağlamaktadır.
Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak Association
Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

Veri Seti Hikayesi:
Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır. Alınan her hizmetin tarih ve saat
bilgisini içermektedir.

4 Değişken 162.523 Gözlem 5 MB
UserId: Müşteri numarası
ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
(Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
CreateDate: Hizmetin satın alındığı tarih
"""

# Görev 1: Veriyi Hazırlama
import pandas as pd
import datetime as dt
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option("display.max_columns", None)

# Adım 1: armut_data.csv dosyasını okutunuz.
df = pd.read_csv("recommendation_systems/armut_data.csv")
df.head()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.\
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri
# başka bir sepeti ifade etmektedir.
# Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
df["CreateDate"].dtype
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()
df["SepetId"] = [str(row[0]) + "_" + row[5] for row in df.values]

# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz.
# Adım 1: Sepet, hizmet pivot table’i oluşturunuz.
df_pivot = df.groupby(["SepetId", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x : 1 if x > 0 else 0)
df_pivot.head()

# Adım 2: Birliktelik kurallarını oluşturunuz.
frequent_itemsets = apriori(df_pivot, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

# Adım3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0")