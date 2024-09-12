# Relatório de Análise de Classificação de Texto

## 1. Introdução

Neste projeto, construímos um modelo de classificação de texto para categorizar reclamações de consumidores em diferentes tipos de produtos. Utilizamos um pipeline que inclui vetorização de texto, transformação TF-IDF e o classificador Naive Bayes Multinomial.

## 2. Pré-processamento dos Dados

Para preparar os dados, realizamos as seguintes etapas:
1. **Leitura do arquivo CSV**: Carregamos o conjunto de dados contendo as colunas `product` e `narrative`.
2. **Pré-processamento do texto**: Aplicamos a tokenização e remoção de stopwords usando NLTK para limpar e normalizar o texto.

```python
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

# Função para pré-processamento de texto
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Carregar o arquivo CSV
url = 'https://drive.google.com/uc?id=1Yt8X9u62TWW-dkH4Kn-XYo4LegUcX_Ix'
data = pd.read_csv(url)
data['cleaned_text'] = data['narrative'].apply(preprocess_text)
```

## 3. Divisão dos Dados

Dividimos o conjunto de dados em conjuntos de treino e teste para avaliar o desempenho do modelo:

```python
X = data['cleaned_text']
y = data['product']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 4. Construção e Treinamento do Modelo

Criamos um pipeline que inclui a vetorização do texto, a transformação TF-IDF e o classificador Naive Bayes Multinomial. O modelo foi treinado com os dados de treino:

```python
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
text_clf.fit(X_train, y_train)
```

## 5. Avaliação do Modelo

O modelo foi avaliado utilizando o conjunto de teste. Calculamos a acurácia, a matriz de confusão e o relatório de classificação. A matriz de confusão foi visualizada com um heatmap.

```python
y_pred = text_clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_report = metrics.classification_report(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=text_clf.classes_, yticklabels=text_clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

## 6. Palavras Importantes para Cada Categoria

Identificamos as palavras mais importantes para cada categoria com base nas log-probabilidades das palavras:

```python
feature_names = text_clf.named_steps['vect'].get_feature_names_out()
log_probs = text_clf.named_steps['clf'].feature_log_prob_

for i, category in enumerate(text_clf.classes_):
    top_indices = log_probs[i].argsort()[-10:]
    top_words = [feature_names[j] for j in top_indices]
    print(f'Categoria: {category}')
    print(f'Palavras principais: {top_words}')
    print()
```

## 7. Conclusão

O modelo de classificação de texto baseado em Naive Bayes Multinomial foi treinado e avaliado com sucesso. As métricas de desempenho mostram que o modelo é capaz de categorizar corretamente as reclamações de consumidores. Além disso, identificamos as palavras mais importantes para cada categoria, o que pode ajudar a entender melhor os tópicos associados a cada produto.

---
