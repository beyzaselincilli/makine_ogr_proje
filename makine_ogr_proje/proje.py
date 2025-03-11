#################################################
########Gerekli Kütüphane ve Fonksiyonlar########
#################################################
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score,validation_curve
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np



pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 170) 
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3F' % x)
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter (action='ignore', category=FutureWarning) 
warnings.simplefilter (action="ignore", category=ConvergenceWarning)
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings(action="ignore", category=SettingWithCopyWarning)

#################################################
# GELİŞMİS FONKSİYONEL KEŞİFCİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#################################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables) You, 12 minutes 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 4. hedef değişken analizi
# 5. korelasyon analizi


#################################################
#GENEL RESİM
#################################################

def check_df(dataframe, head=5):
    print("################## Shape ###################")
    print(dataframe.shape)
    
    print("\n################ Types ####################")
    print(dataframe.dtypes)
    
    print("\n################# Head ###################")
    print(dataframe.head(head))
    
    print("\n############## Tail ################")
    print(dataframe.tail(head))
    
    print("\n############## NA ####################")
    print(dataframe.isnull().sum())
    
    print("\n############## Quantiles ####################")
    print(dataframe.describe([0,0.05,0.5,0.95,0.99,1]).T)

# Örnek CSV dosyası yükleme
df = pd.read_csv("C:/Users/User/.spyder-py3/proje/hitters.csv")



############### Fonksiyonu çağırma ###############
check_df(df)





def show_salary_skewness(df):
    # Salary sütunundaki eksik değerleri kaldırma
    salary = df['Salary'].dropna()

    # Çarpıklık değerini hesaplama
    skewness = salary.skew()

    # Histogram ve yoğunluk grafiği çizme
    plt.figure(figsize=(12, 6))
    sns.histplot(salary, kde=True)
    plt.title(f"Salary Distribution (Skewness: {skewness:.2f})", fontsize=16)
    plt.xlabel("Salary (thousands)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Ortalama ve medyanı gösterme
    plt.axvline(salary.mean(), color='r', linestyle='dashed', linewidth=2, label=f"Mean: {salary.mean():.2f}")
    plt.axvline(salary.median(), color='g', linestyle='dashed', linewidth=2, label=f"Median: {salary.median():.2f}")
    plt.legend()

    # Grafik gösterimi
    plt.show()

    # İstatistiksel bilgileri yazdırma
    print(f"Skewness: {skewness:.2f}")
    print(f"Mean: {salary.mean():.2f}")
    print(f"Median: {salary.median():.2f}")
    print(f"Standard Deviation: {salary.std():.2f}")
    print(f"Minimum: {salary.min():.2f}")
    print(f"Maximum: {salary.max():.2f}")
    
    #shapiro-wilk normallik testi
    _,p_value=stats.shapiro(salary)
    print(f"Shapiro-wilk Test p-value: {p_value:.4f}")
    plt.show(block=True)
    
############### Fonksiyonu çağırma ###############
show_salary_skewness(df)
#verilerde sağa doğru bir çarpıklık var


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    #Tüm kategorik değişkenleri tutar.
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and 
                               dataframe[col].dtypes != "O"]
    #Sayısal ama kategorik gibi davranan değişkenleri tutar (benzersiz değer sayısı cat_th'dan az).
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                               dataframe[col].dtypes == "O"]
    #Kategorik ama sayısal gibi davranan değişkenleri tutar (benzersiz değer sayısı car_th'dan fazla).
    
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#################################################
#KATEGORİK DEĞİŞKEN ANALİZİ
#################################################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100* dataframe [col_name].value_counts() / len(dataframe)}))
    print("######################################################")
  
    if plot:
        sns.countplot(x=dataframe [col_name], data=dataframe) 
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

#################################################
#SAYISAL DEĞİŞKEN ANALİZİ
#################################################
def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] 
    print(dataframe [numerical_col].describe (quantiles).T)
    if plot:
        dataframe [numerical_col].hist (bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
        
for col in num_cols:
    num_summary(df, col, plot=True)
#aykırı değerler burada görülebilir

#################################################
#HEDEF DEĞİŞKEN ANALİZİ
#################################################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}),end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

#################################################
#KORELASYON ANALİZİ
#################################################
df[num_cols].corr(method="spearman")

fig, ax = plt.subplots (figsize=(25,10))
sns.heatmap (df [num_cols].corr(), annot=True, linewidths=.5, ax=ax)
plt.show(block=True)


def find_correlation(dataframe, numeric_cols, corr_limit=0.50):
    high_correlations = [] 
    low_correlations = [] 
    for col in numeric_cols: 
        if col == "Salary":    
            pass
        else:
            correlation = dataframe[[col, "Salary"]].corr().loc[col, "Salary"] 
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ":" + str(correlation)) 
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs=find_correlation(df, num_cols)
#parametredeki corr_limit değeri 0.6 olduğunda yüksek korelasyon bulunmuyor
#bundan kaynaklı corr_limit değerini 0.5 yaptık

#################################################
#OUTLIERS(AYKIRI DEĞERLER)
#################################################

#hedef değişken
sns.boxplot(x=df["Salary"], data=df)
plt.show(block=True)

#sayısal değişkenler
for col in df[num_cols]:
    sns.boxplot(x=df[col],data=df)
    plt.show(block=True)
    
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
#neden 0.1-0.9 yaptık? çünkü cattbat, chmrun, crns,uncommitted changes değerlerindeki aykırılık çok yüksekti
#ancak kontrol 0.1-0.9 ile aykırı değerler kontrol edildiğinde hala dışarıda kalan aykırı değerler vardı bundan dolayı 0.25-0.75 olarak(varsayılan olarak) durumu güncelledik

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    
    # bir aykırılık var mı? kontrol ediyoruz
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    
    # Değerleri düşük limitten küçük olanları düşük limit ile, yüksek limitten büyük olanları ise yüksek limit ile değiştir
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# num_cols sütunlarında outlier kontrolü yap ve sonuçları yazdır
for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


#################################################
#MISSING VALUES(EKSİK DEĞERLER)
#################################################

def missing_values_table(dataframe, na_name=False):
    # NaN (eksik) değerlere sahip olan sütunları bul
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    
    # Eksik değerlerin sayısını al
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    
    # Eksik değer oranını hesapla
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    
    # Sonuçları birleştir ve bir DataFrame oluştur
    missing_df = pd.concat([n_miss, np.round(ratio, decimals=2)], axis=1, keys=['n_miss', 'ratio'])
    
    # Eksik değerlerin sayısı ve oranını yazdır
    print(missing_df, end="\n")
    
    # Eğer na_name True ise, eksik değerler olan sütunları döndür
    if na_name:
        return na_columns

# Fonksiyonu çağır
missing_values_table(df)


# Eksik veri analizi için 3 farklı yöntem kullanılabilir.
# df1'in bir kopyasını oluştur
df1 = df.copy()

# df1'in ilk birkaç satırını görüntüle
df1.head()

# Kategorik ve sayısal sütunları, ayrıca kategorik fakat çok değerli (cardinality) sütunları ayır
cat_cols, num_cols, cat_but_car = grab_col_names(df1)




# Eksik veri doldurma fonksiyonu
def eksik_veri_doldur(dataframe, method):
    # DataFrame'in bir kopyasını al
    df1 = dataframe.copy()

    # Kategorik ve sayısal sütunları ayırmak için fonksiyon çağrısı
    cat_cols, num_cols, cat_but_car = grab_col_names(df1)
    
    # Eğer method 1 seçildiyse, KNNImputer ve RobustScaler ile eksik verileri doldur
    if method == 1:
        # Kategorik ve sayısal sütunları birleştir
        dff = pd.get_dummies(df1[cat_cols + num_cols], drop_first=True)

        # Veriyi ölçeklendir
        scaler = RobustScaler()
        dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

        # KNNImputer ile eksik verileri doldur
        imputer = KNNImputer(n_neighbors=5)
        dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

        # Ölçeklendirilmiş veriyi orijinal ölçeğe geri döndür
        dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
        
        # Sonuçları df1'e geri aktar
        df1 = dff

    # Eğer method 2 seçildiyse, gruplama ve ortalama ile eksik verileri doldur
    elif method == 2:
        # "League" ve "Division" sütunlarına göre gruplama yapıp, ortalama değer ile eksik "Salary" değerlerini dolduruyoruz
        # "League" = "A" ve "Division" = "E" olan gruptaki eksik "Salary" değerlerini doldur
        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "A") & (df1["Division"] == "E"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].transform("mean")["A", "E"]

        # "League" = "A" ve "Division" = "W" olan gruptaki eksik "Salary" değerlerini doldur
        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "A") & (df1["Division"] == "W"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].transform("mean")["A", "W"]

        # "League" = "N" ve "Division" = "E" olan gruptaki eksik "Salary" değerlerini doldur
        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "N") & (df1["Division"] == "E"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].transform("mean")["N", "E"]

        # "League" = "N" ve "Division" = "W" olan gruptaki eksik "Salary" değerlerini doldur
        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "N") & (df1["Division"] == "W"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].transform("mean")["N", "W"]

    # Eğer method 3 seçildiyse, eksik veri içeren tüm satırları sil
    elif method == 3:
        # Eksik değer içeren tüm satırları silme
        df1.dropna(inplace=True)
    
    return df1


df1=eksik_veri_doldur(df, method=1)

print(df1.head())
print(df1.isnull().sum())


#################################################
#FEATURE EXTRACTION(ÖZELLİK ÇIKARIMI)
#################################################

# Num_cols listesinden "Salary" sütununu çıkarıyoruz
new_num_cols = [col for col in num_cols if col != "Salary"]

# DataFrame'deki yeni sayısal sütunlara 0.0000000001 ekliyoruz (nümerik hesaplamaları iyileştirme için)
df1[new_num_cols] = df1[new_num_cols] + 0.0000000001

# Yeni sütunlar ekliyoruz ve hesaplamaları gerçekleştiriyoruz

# Hits'in AtBat'a oranı, vuruş başarı oranını hesaplamak için oluşturulmuş bir özellik
df1["Hits_Success"] = (df1["Hits"] / df1["AtBat"]) * 100

# RBI / CRBI oranı, oyuncunun ne kadar verimli koşu yaptırdığına dair bir gösterge sağlar
df1["NEW_RBI"] = df1["RBI"] / df1["CRBI"]

# Walks / cwalks oranı, oyuncunun ligdeki genel başarısını ölçmek için kullanılır
df1["NEW_Walks"] = df1["Walks"] / df1["CWalks"]

# Putouts * Years, oyuncunun yıllık başarı gösterisini belirler
df1["NEW_PutOuts"] = df1["PutOuts"] * df1["Years"]

# Hits / CHits + Hits, oyuncunun kariyerindeki hit başarı oranını artırmak için yeni bir özellik
df1["NEW_Hits"] = (df1["Hits"] / df1["CHits"]) + df1["Hits"]

# CRBI * CAtBat, oyuncunun kariyerindeki toplam verimlilikle ilgili bir özellik
df1["NEW_CRBI*CATBAT"] = df1["CRBI"] * df1["CAtBat"]

# CHits / Years, oyuncunun kariyerine göre yıllık ortalama vuruş başarısını gösterir
df1["NEW_Chits"] = df1["CHits"] / df1["Years"]

# CHmRun * Years, oyuncunun kariyerindeki yıllık değerli vuruş performansını gösterir
df1["NEW_CHmRun"] = df1["CHmRun"] * df1["Years"]

# CRuns / Years, oyuncunun kariyerine göre yıllık kazandırdığı sayıyı ölçer
df1["NEW_CRuns"] = df1["CRuns"] / df1["Years"]

# NEW_RW, RBI ve Walks arasında bir ilişkiyi gösterir; oyuncunun verimliliğine dair bir gösterge sunar
df1["NEW_RW"] = df1["RBI"] * df1["Walks"]

# CH_CB, RBI / Walks ve CHits / CAtBat arasındaki ilişkiyi gösterir; oyuncunun oyun tarzının birleşimi hakkında bilgi verir
df1["NEW_CH_CB"] = (df1["RBI"] / df1["Walks"]) * (df1["CHits"] / df1["CAtBat"])

# CHmRun / CAtBat, oyuncunun kariyerindeki her at-bat için kazandığı değerli vuruş sayısını ölçer
df1["NEW_CHmRun_CAtBat"] = df1["CHmRun"] / df1["CAtBat"]

# NEW_Diff_Atbat, oyuncunun yıllık at-bat sayısı ile toplam kariyer at-bat sayısı arasındaki farkı gösterir
df1['NEW_Diff_Atbat'] = df1['AtBat'] - (df1['CAtBat'] / df1['Years'])

# NEW_Diff_Hits, oyuncunun yıllık vuruş sayısı ile kariyer vuruş sayısı arasındaki farkı gösterir
df1['NEW_Diff_Hits'] = df1['Hits'] - (df1['CHits'] / df1['Years'])

# NEW_Diff_HmRun, oyuncunun yıllık en değerli vuruş sayısı ile kariyer en değerli vuruş sayısı arasındaki farkı gösterir
df1['NEW_Diff_HmRun'] = df1['HmRun'] - (df1['CHmRun'] / df1['Years'])

# NEW_Diff_Runs, oyuncunun yıllık kazandırdığı koşu sayısı ile kariyer kazandırdığı koşu sayısı arasındaki farkı gösterir
df1['NEW_Diff_Runs'] = df1['Runs'] - (df1['CRuns'] / df1['Years'])

# NEW_Diff_RBI, oyuncunun yıllık koşu yaptırdığı oyuncu sayısı ile kariyer koşu yaptırdığı oyuncu sayısı arasındaki farkı gösterir
df1['NEW_Diff_RBI'] = df1['RBI'] - (df1['CRBI'] / df1['Years'])

# NEW_Diff_walks, oyuncunun yıllık hata sayısı ile kariyer hata sayısı arasındaki farkı gösterir
df1['NEW_Diff_Walks'] = df1['Walks'] - (df1['CWalks'] / df1['Years'])



df1["Salary"].isnull().sum()

#################################################
#ONE-HOT ENCODİNG
#################################################

cat_cols, num_cols,cat_but_car=grab_col_names(df1)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False): #drop_first yapılmasının sebebi dummy tuzağına düşmemek için kullanılır.
    # Verilen kategorik sütunları one-hot encoding işlemi ile dönüştür
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    
    # Bool sütunlarını 0 ve 1'e dönüştür
    for col in dataframe.columns:
        if dataframe[col].dtype == 'bool':
            dataframe[col] = dataframe[col].astype(int)
    
    return dataframe


df1 = one_hot_encoder(df1, cat_cols, drop_first=True)

df1.isnull().sum().sum()




#################################################
#FEATURE SCALİNG (ÖZELLİK ÖLÇEKLENDİRME)
#################################################


cat_cols, num_cols,cat_but_car=grab_col_names(df1)


num_cols = [col for col in num_cols if col not in ["Salary"]]

def feature_scaling(dataframe, num_cols):
    # StandardScaler nesnesini oluştur
    scaler = StandardScaler()
    cat_cols,num_cols,cat_but_car=grab_col_names(dataframe)
    # "Salary" sütununu num_cols listesinden çıkar
    num_cols = [col for col in num_cols if col not in ["Salary"]]

    # Num_cols için ölçeklendirme işlemi yap
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe


df1[num_cols].corr(method="spearman")

# Correlation Analysis
fig, ax = plt.subplots(figsize=(25, 10))  # Grafik boyutlarını ayarla
sns.heatmap(df1.corr(), annot=True, linewidths=0.5, ax=ax)  # Isı haritası oluştur
plt.title("Correlation Heatmap", fontsize=18)  # Grafik başlığı ekle
plt.show()  # Grafiği göster

# Correlation matrix
corr_matrix = df1.corr()  # Korelasyon matrisi hesapla
corr_matrix  # Matrisin çıktısını göster


# Yüksek korelasyon eşik değeri
high_corr_threshold = 0.95

# Yüksek korelasyona sahip özelliklerin seti
high_corr_features = set()

# Korelasyon matrisinde gez
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)

# Sonuçları yazdır
high_corr_features
"""
yüksek korelasyonlu özellikler çıkartılarsa modelde daha başarılı bir sonuç elde edilebilir
"""


#################################################
#MODELİNG
#################################################


# Eksik değerleri kaldır
df1.dropna(inplace=True)

# Kalan eksik değerleri kontrol et
df1.isnull().sum().sum()

# Hedef değişkeni tanımla
y = df1["Salary"]

# Bağımsız değişkenleri tanımla
X = df1.drop(labels=["Salary"], axis=1)

# Veri şekillerini kontrol et
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=46)
#satır sayısı düşük olduğundan kaynaklı test_size değerini 0.2 olarak belirliyoruz


# Doğrusal Regresyon için Model Değerlendirmesi
linreg = LinearRegression()
model=linreg.fit(X_train, y_train)
y_pred = model.predict(X_train)
lin_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("LINEAR REGRESSION TRAIN RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred))))


# Model Evaluation for Linear Regression - Train and Test Scores

# Calculate and print R-squared score for training data
lin_train_r2 = linreg.score(X_train, y_train)
print("LINEAR REGRESSION TRAIN R-SQUARED: ", "{:,.3f}".format(linreg.score(X_train, y_train)))

# Predict on test data
y_pred = model.predict(X_test)

# Calculate and print RMSE for test data
lin_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("LINEAR REGRESSION TEST RMSE:", "{:,.2f}".format( np.sqrt(mean_squared_error(y_test, y_pred))))

"""
LINEAR REGRESSION TRAIN RMSE: 193.32
LINEAR REGRESSION TEST RMSE: 193.60
bu durumda train ve test arasında iyi bir bağlantı olduğunu tespit ediyoruz

LINEAR REGRESSION TRAIN R-SQUARED:  0.751
bağımsız değişkenin bağımlı değişkeni tarif etme durumu
çok iyi olmasa bile az veri ile çalışıldığından dolayı iyi bir oran
"""

# Test part regplot for visualizing predictions vs actual values
g = sns.regplot(
    x=y_test,
    y=y_pred,
    scatter_kws={'color': 'b', 's': 5},
    ci=False,
    color="r"
)
g.set_title(f"Test Model R2: {linreg.score(X_test, y_test):.3f}")
g.set_ylabel("Predicted Salary")
g.set_xlabel("Salary")
plt.xlim(-5, 2700)
plt.ylim(bottom=0)
plt.show(block=True)


# Calculate cross-validation score
print("LINEAR REGRESSION CROSS_VAL_SCORE:", "{:,.3f}".format(np.mean(np.sqrt(-cross_val_score(
    model,
    X,
    y,
    cv=10,
    scoring="neg_mean_squared_error" ))))) 
"""LINEAR REGRESSION CROSS_VAL SCORE: 225.509"""                                                                                        
#bu pek iyi bir yöntem olmaz çünkü verisetimizde sütun sayımız az 

X_train_sm=sm.add_constant(X_train)

model_sm=sm.OLS(y_train, X_train_sm).fit()

model_summary=model_sm.summary()
model_summary




# Veriyi bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Modeli tanımlayıp eğitelim
lgb_model = LGBMRegressor(verbose=-1).fit(X_train, y_train)

# Train Error
y_pred = lgb_model.predict(X_train)

# Test Error
y_pred1 = lgb_model.predict(X_test)

# Train RMSE ve R2
lightgbm_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
lightgbm_train_r2 = r2_score(y_train, y_pred)

# Test RMSE ve R2
lightgbm_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred1))
lightgbm_test_r2 = r2_score(y_test, y_pred1)

# Sonuçları yazdıralım
print("LGBM Train RMSE:", "{:,.2f}".format(lightgbm_train_rmse), "\n")
print("LGBM Test RMSE:", "{:,.2f}".format(lightgbm_test_rmse), "\n")
print("LGBM Train R2:", "{:,.2f}".format(lightgbm_train_r2), "\n")
print("LGBM Test R2:", "{:,.2f}".format(lightgbm_test_r2))

"""
LGBM Train RMSE: 76.59 

LGBM Test RMSE: 213.65 

LGBM Train R2: 0.96 

LGBM Test R2: 0.73
regresyondan daha başarısız bir model çünkü LGBM çok fazla veri isteyen bir modeldir
lineer regresyon ile arasıdna büyük bir fark yok ancak performans olarak daha fazla  
"""

#################################################
#HİPERPARAMETRE
#################################################

# Model Parametreleri(Hiperparametreler)
lgb_model = LGBMRegressor()

lgb_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [500, 1000],
    "max_depth": [3, 5, 8],
    "colsample_bytree": [1, 0.8, 0.5]
}

# GridSearchCV ile en iyi parametreleri bulma
lgb_cv_model = GridSearchCV(
    lgb_model,
    lgb_params,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Modeli eğitme
lgb_cv_model.fit(X_train, y_train)

# En iyi parametreleri yazdırma
print("Best Parameters:", lgb_cv_model.best_params_)




# LightGBM Modeli ile Eğitim (Hyperparametre Ayarlarıyla)
lgb_tuned = LGBMRegressor(**lgb_cv_model.best_params_).fit(X_train, y_train)

# Test Verisi ile Tahmin Yapma
y_pred = lgb_tuned.predict(X_test)

# Modelin Performansını Değerlendirme
lightgbm_tuned_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lightgbm_tuned_test_r2 = r2_score(y_test, y_pred)
lightgbm_tuned_test_mae = mean_absolute_error(y_test, y_pred)

# Sonuçları Yazdırma
print("\nLightGBM Tuned Test RMSE:", "{:,.2f}".format(lightgbm_tuned_test_rmse), "\n")
print("LightGBM Tuned Test MAE:", "{:,.2f}".format(lightgbm_tuned_test_mae), "\n")
print("LightGBM Tuned Test R^2:", "{:,.2f}".format(lightgbm_tuned_test_r2), "\n")


# Feature Importance Fonksiyonu
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:num])
    
    plt.title('Features Importance')
    plt.tight_layout()
    plt.show()
    
    if save:
        plt.savefig('importances.png')

# Feature Importance Görselleştirme
plot_importance(lgb_tuned, X_test)





# Veri ve hedef değişkenin ayrılması
y = df1["Salary"]
X = df1.drop("Salary", axis=1)

# Eğitim ve test verilerinin bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Modelin eğitilmesi
rf_model = RandomForestRegressor().fit(X_train, y_train)

# Train Error
y_pred = rf_model.predict(X_train)

# Test Error
y_pred1 = rf_model.predict(X_test)

# Train RMSE ve Test RMSE
rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred1))

# Train R^2 ve Test R^2
rf_train_r2 = r2_score(y_train, y_pred)
rf_test_r2 = r2_score(y_test, y_pred1)

# Sonuçların yazdırılması
print("RF Train RMSE:", "{:,.2f}".format(rf_train_rmse))
print("RF Test RMSE:", "{:,.2f}".format(rf_test_rmse), "\n")
print("RF Train MAE:", "{:,.2f}".format(mean_absolute_error(y_train, y_pred)))
print("RF Test MAE:", "{:,.2f}".format(mean_absolute_error(y_test, y_pred1)), "\n")
print("RF Train R^2:", "{:,.2f}".format(rf_train_r2))
print("RF Test R^2:", "{:,.2f}".format(rf_test_r2))

# Model Parametreleri
rf_params = {
    "max_depth": [5, 8, None],
    "max_features": [3, 5, 15],
    "n_estimators": [200, 500],
    "min_samples_split": [2, 5, 8]
}

# GridSearchCV ile en iyi parametreleri bulma
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1)

# Modeli eğitme
rf_cv_model.fit(X_train, y_train)

# En iyi parametreleri yazdırma
print(rf_cv_model.best_params_)


# RF TUNED Model
rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)

# Tahminler
y_pred = rf_tuned.predict(X_test)

# Test Error (RMSE ve R^2)
rf_tuned_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rf_tuned_test_r2 = r2_score(y_test, y_pred)

# Sonuçları Yazdırma
print("RF_TUNED Test RMSE:", "{:,.2f}".format(rf_tuned_test_rmse), "\n")
print("RF_TUNED Test MAE:", "{:,.2f}".format(mean_absolute_error(y_test, y_pred)), "\n")
print("RF_TUNED Test R^2:", "{:,.2f}".format(rf_tuned_test_r2))


# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[0:num])
    plt.title('Features Importance')
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig('importances.png')

plot_importance(rf_tuned, X_test)

