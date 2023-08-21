import pandas as pd
import streamlit as st
import re
import nltk
import scipy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.sparse import csr_matrix

#Menampilkan judul
st.title('Sentimen Analisis Twitter'
         ' Pemberlakuan Pembatasan Kegiatan Masyarakat (PPKM)')

#Membaca dan menampilkan data
df = pd.read_csv('Clean_Data_3Class.csv', sep=';')
st.write('### Data Awal')
st.dataframe(df)

# Fungsi untuk pre-processing data
def preprocess(text):
  # Remove special characters and digits
  text = re.sub(r'[^a-zA-Z]', ' ', str(text))

  # Convert to lowercase
  text = text.lower()

  # Tokenize the text
  tokens = nltk.word_tokenize(text)

  # Remove stop words
  stop_words = set(stopwords.words('indonesian'))
  tokens = [w for w in tokens if w not in stop_words]

  # Perform stemming
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  tokens = [stemmer.stem(tokens) for tokens in tokens]

  return ' '.join(tokens)  # Menggabungkan kembali kata-kata menjadi kalimat


class MultinomialNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_probs = None
        self.word_probs = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        self.class_probs = np.zeros(num_classes)
        self.word_probs = np.zeros((num_classes, X.shape[1]))  # Gunakan matriks numpy

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            total_word_count_c = X_c.getnnz()
            self.class_probs[i] = X_c.shape[0] / X.shape[0]

            word_probs_c = (X_c.sum(axis=0) + 1) / (total_word_count_c + X.shape[1])
            self.word_probs[i] = word_probs_c  # Simpan probabilitas kata dalam matriks numpy

    def predict(self, X):
        preds = []

        for doc in X:
            log_probs = np.log(self.class_probs)

            for i, word_count in enumerate(doc.toarray()[0]):
                for j, word_prob in enumerate(self.word_probs):
                    log_probs[j] += word_count * np.log(word_prob[i])

            pred = self.classes[np.argmax(log_probs)]
            preds.append(pred)

        return preds

    def score(self, X, y):
        # Menghitung akurasi model pada data X dengan label y
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# mengaplikasikan fungsi preprocess pada kolom Text
df['Text'] = df['Text'].apply(preprocess)
st.write('### Data setelah pre-processing')
st.dataframe(df)

feature = df['Text'].astype(str)
target = df['Sentimen'].astype(str)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.25,
                                                    random_state=123)

# Melakukan vektorisasi pada data latih
tf = TfidfVectorizer()
tf_data = tf.fit_transform(X_train)

data_vlatih = pd.DataFrame(tf_data.toarray(),columns=tf.get_feature_names_out())
st.write('### Data latih setelah vektorisasi')
st.dataframe(data_vlatih)

#Membuat model klasifikasi NBC
mnb = MultinomialNaiveBayes()
mnb.fit(tf_data,y_train)

st.write('### Akurasi menggunakan data latih')
st.write(mnb.score(tf_data,y_train))


#Membuat visualisasi confusion matrix untuk data latih
pred = mnb.predict(tf_data)
cm = confusion_matrix(pred,y_train)
ConfusionMatrixDisplay(cm,display_labels=['negatif','netral','positif']).plot()
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
st.write('### Confusion Matrix menggunakan data latih')
st.pyplot(plt)

#Melakukan vektorisasi pada data uji dan menampilkannya
test_data = tf.transform(X_test)
data_vuji = pd.DataFrame(test_data.toarray(),columns=tf.get_feature_names_out())
st.write('### Data uji setelah vektorisasi')
st.dataframe(data_vuji)

#menampilkan akurasi menggunakan data uji
st.write('### Akurasi menggunakan data uji')
st.write(mnb.score(test_data,y_test))

#Membuat visualisasi confusion matrix untuk data uji
pred = mnb.predict(test_data)
cm = confusion_matrix(pred,y_test)
ConfusionMatrixDisplay(cm,display_labels=['negatif','netral','positif']).plot()
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
st.write('### Confusion Matrix menggunakan data uji')
st.pyplot(plt)

# user_text = st.text_input('Text', '')
#
# # if st.button('Classify'):
# #   user_text = preprocess(user_text)
# #   vect = TfidfVectorizer()
# #   pred_data = tf.transform(user_text)
# #   prediksi = mnb.predict(pred_data)
# #   st.write(prediksi)
