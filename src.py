import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

# Baixar recursos adicionais do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Função para pré-processamento de texto
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    # Tokenização e remoção de stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Carregar o arquivo CSV
url = 'https://drive.google.com/uc?id=1Yt8X9u62TWW-dkH4Kn-XYo4LegUcX_Ix'
data = pd.read_csv(url)

# Exibir as primeiras linhas do DataFrame e as colunas
print("Primeiras linhas do DataFrame:")
print(data.head())
print("\nColunas do DataFrame:")
print(data.columns)

# Aplicar pré-processamento no texto
data['cleaned_text'] = data['narrative'].apply(preprocess_text)

# Dividir dados em treino e teste
X = data['cleaned_text']
y = data['product']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

text_clf.fit(X_train, y_train)

# Fazer predições
y_pred = text_clf.predict(X_test)

# Avaliar o modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_report = metrics.classification_report(y_test, y_pred)

# Mostrar métricas
print(f'\nAcurácia: {accuracy:.2f}')
print('\nMatriz de Confusão:')
print(conf_matrix)
print('\nRelatório de Classificação:')
print(class_report)

# Visualizar matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=text_clf.classes_, yticklabels=text_clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Identificar palavras importantes para cada categoria
feature_names = text_clf.named_steps['vect'].get_feature_names_out()
log_probs = text_clf.named_steps['clf'].feature_log_prob_

for i, category in enumerate(text_clf.classes_):
    top_indices = log_probs[i].argsort()[-10:]  # Top 10 palavras
    top_words = [feature_names[j] for j in top_indices]
    print(f'Categoria: {category}')
    print(f'Palavras principais: {top_words}')
    print()
