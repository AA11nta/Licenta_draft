from tkinter import Image

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from lime import lime_tabular
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import lime
import numpy as np
import pandas as pd
import qrcode
import seaborn as sns
import streamlit as st

apptitle = 'Application'
st.set_page_config(page_title=apptitle, page_icon=":bar_chart:")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-color: #F60360 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def get_data():
    return pd.read_csv('https://raw.githubusercontent.com/AA11nta/try/main/BreastCancerProject/breast-cancer.csv', header=0)


# def get_data_filter():
#     return pd.read_csv('https://raw.githubusercontent.com/tipemat/datasethoracic/main/DateToracic.csv', header=0)

#
# def codQR():
#     link = "https://cflavia-dizertatie-main-nhrwqh.streamlit.app/"  # Link-ul către aplicație
#
#     qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
#     qr.add_data(link)
#     qr.make(fit=True)
#
#     qr_img = qr.make_image(fill_color="black", back_color="white")
#     qr_img.save("qrcode.png")


# def load_data_map():
#     data = pd.read_csv('resources/covid.csv')
#     data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
#     return data


# def get_data_predict():
#     return pd.read_csv('resources/diabetes_data_upload.csv')


data = get_data()
st.sidebar.subheader("Prezentare generală a aplicației")
btn_afis_general = st.sidebar.button("Introducere")

choose_tabel = st.sidebar.button("Setul de date")
if (choose_tabel):
    st.subheader("Setul de date")
    st.write(
        "Setul de date Breast Cancer de pe Kaggle, furnizat de Yasser Hesham, include informații despre diverse caracteristici ale tumorilor mamare."
        
        "Setul de date este utilizat pentru clasificare binară pentru a prezice dacă o tumoră este malignă sau benignă. Caracteristicile includ atribute precum raza, textura, perimetrul, aria, netezimea și altele.")
    df = get_data()
    # df = df.drop(columns="No")
    st.dataframe(df, height=450, hide_index=True)

    st.write("\n"
             "Setul de date conține informații despre 569 cazuri de cancer mamar. Fiecare caz este descris prin 30 de caracteristici diferite, care analizează trăsăturile nucleilor celulari prezenți în imaginea digitalizată a unei aspirate fine cu ac (FNA) a unei mase mamare. ")
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x='diagnosis', data=df, palette='hls')
    st.pyplot(fig)

    # df.rename(columns={'Sem0l1 ': 'Sem0l1'}, inplace=True)

    for col in df.columns:
        df.loc[(df["diagnosis"] == 0) & (df[col].isnull()), col] = df[df["diagnosis"] == 0][col].median()
        df.loc[(df["diagnosis"] == 1) & (df[col].isnull()), col] = df[df["diagnosis"] == 1][col].median()
    st.write(
        "<div style='text-align:justify;font-size: 16px;'>Mai jos puteți vizualiza histograma cu valorile pentru fiecare dintre componentele care influențează diagnosticul."
        "<li>Cu cât punctele sunt mai răspândite, cu atât valorile sunt mai diverse. Locurile în care sunt strâns legate indică faptul că valorile respective sunt apropiate, reprezentând o majoritate.</li>"
        "<li style='color: red'>diagnosis: 0.0 - Persoane care au tumoare maligna</li>"
        "<li style='color: blue'>diagnosis: 1.0 - Persoane care au tumoare beligna</li></div>",
        unsafe_allow_html=True)
    fig, axes = plt.subplots(9, 1, figsize=(20, 5))

    for col in df.columns:
        if col != "diagnosis":
            st.write('\n')
            sns.catplot(x="diagnosis", y=col, data=df, hue='diagnosis', palette=sns.color_palette(['red', 'blue']),
                        height=5.5, aspect=11.7 / 6.5, legend=True)
            st.write("Grafic de dispersie pentru " + col + "  în ambele cazuri: tumoare maligna sau beligna. ")
            st.pyplot()

    scaler = MinMaxScaler()
    fig = plt.figure(figsize=(20, 15))
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")

    plt.title("Correlation Matrix")
    plt.xlabel("Variable")
    plt.ylabel("Variable")
    plt.show()
    st.pyplot(fig)

if btn_afis_general:
    st.title("Analiza Seturilor de Date pentru Cancerul Mamar: Abordări Avansate cu Inteligență Artificială")
    st.write(

        "Introducere: În ultimii ani, analiza seturilor de date medicale a devenit o componentă esențială în domeniul sănătății, oferind posibilitatea de a îmbunătăți diagnosticarea și tratamentul bolilor. Cancerul mamar, fiind una dintre cele mai comune forme de cancer la femei, beneficiază semnificativ de pe urma acestor tehnologii avansate."
        "\n\n"
        "Metode: Această lucrare prezintă un studiu comparativ între algoritmii de reducere a dimensionalității pentru analiza seturilor de date referitoare la cancerul mamar. În special, se analizează utilizarea Analizei Componentelor Principale (PCA) și un alt algoritm avansat de inteligență artificială. Studiul urmărește să identifice avantajele și dezavantajele fiecărei metode, bazându-se pe experimentele realizate. Rezultatele obținute subliniază eficiența fiecărei abordări în contextul diagnosticării cancerului mamar."
        "\n\n"
        "Rezultate: Precizia modelelor antrenate în acest mod a atins valoarea de peste 80% pentru ambele sarcini clinice."
        "\n"
        "\n"
        "Concluzii: //todo"
    )

choose_MecanismAtentie = st.sidebar.button("Algoritmi existenti")
if choose_MecanismAtentie:
    st.header("PCA (Analiza Componentelor Principale)")
    st.write("**1) PCA detalii**")

    with st.expander("Definirea algoritmului:"):
        st.write("- **PCA (Principal Component Analysis)** este o tehnică de reducere a dimensiunii datelor folosită în analiza statistică și în învățarea automată. Scopul principal al PCA este de a transforma un set mare de variabile corelate într-un set mai mic de variabile necorelate, numite componente principale.")
        st.write("- Folosirea PCA (Principal Component Analysis) oferă multiple **avantaje** în analiza datelor și învățarea automată. În primul rând, reduce dimensiunea datelor păstrând variabilitatea esențială, ceea ce simplifică analiza și vizualizarea și reduce timpul și resursele computaționale necesare. De asemenea, elimină redundanța informațiilor prin combinarea variabilelor corelate în componente principale necorelate, rezultând modele mai eficiente și mai rapide. PCA îmbunătățește performanța algoritmilor de învățare automată, reducând riscul de overfitting și crescând acuratețea modelelor. Facilitează vizualizarea datelor multidimensionale în două sau trei dimensiuni, ajutând la identificarea tiparelor, grupurilor sau outlier-ilor. Permite compresia datelor, economisind spațiu de stocare și păstrând informațiile relevante. PCA este utilizată ca un pas de preprocesare pentru a îmbunătăți calitatea datelor și a facilita antrenarea modelelor și ajută la reducerea zgomotului, extrăgând componentele principale care reprezintă cele mai semnificative variații, îmbunătățind claritatea și relevanța datelor. Aceste avantaje fac din PCA un instrument valoros în analiza datelor și dezvoltarea modelelor predictive, mai ales pentru seturile de date mari și complexe.")
    st.write("**Rezultatele obtinute dupa aplicarea algoritmului PCA**")
    st.write("- Așa cum se poate observa din diagrama de mai jos, există caracteristici care contribuie semnificativ la "
             "obținerea unei predicții mai bune pentru cancerul mamar.")
    df = get_data()
    # Separate features and target
    X = df.drop(columns=["diagnosis"])
    Y = df["diagnosis"]

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Convert the results to a DataFrame
    X_pca_df = pd.DataFrame(data=X_pca, columns=['Componenta Principala 1', 'Componenta Principala 2'])

    # Display explained variance ratio
    st.write("Variatia explicata de fiecare componenta principala:")
    st.write(pca.explained_variance_ratio_)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Apply PCA on the training set
    X_train_pca = pca.fit_transform(X_train)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_pca, y_train)

    # Apply PCA on the test set
    X_test_pca = pca.transform(X_test)

    # Calculate classification probabilities for the test set
    probs = model.predict_proba(X_test_pca)
    # Get the probability associated with class 1
    preds = probs[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # Calculate predictions on the test set
    y_pred = model.predict(X_test_pca)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display confusion matrix
    st.write("Matricea de confuzie:")
    st.write(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

    st.write("**2) LDA (Linear Discriminant Analysis)**")

    with st.expander("Descrierea algoritmului"):
        st.write("- Este un algoritm de reducere a dimensiunii și de clasificare care maximizează separabilitatea dintre clase.")

    # with st.expander("The description of the algorithm used."):
    #     st.write(
    #         "- The developed prediction model consists of two types of layers, namely: **Dense și BatchNormalization**")
    #     st.write(
    #         "- The **Dense** layer is used in neural networks, where each neuron in the current layer connects to all "
    #         "neurons in the next layer. It is a fully connected layer, where each neuron receives all input values from "
    #         "the previous layer and produces an output value.")
    #     st.write(
    #         "- The **BatchNormalization** layer is a layer used in deep neural networks to normalize activations"
    #         " between layers during the training process. It was introduced to help speed up training, "
    #         "reduce overfitting, and improve the model's generalization.")
    # st.write("- After training the model, the following results were achieved:")

    # Separate features and target
    X = df.drop(columns=["diagnosis"])
    Y = df["diagnosis"]

    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X, Y)

    # Convert the results to a DataFrame
    X_lda_df = pd.DataFrame(data=X_lda, columns=['Componenta LDA 1'])

    # Plot LDA
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_lda_df['Componenta LDA 1'], [0] * len(X_lda_df), c=Y, cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Diagnostice")
    ax.add_artist(legend)
    plt.xlabel('Componenta LDA 1')
    plt.ylabel('Valoare')
    plt.title('Proiectarea LDA a datelor')
    st.pyplot(fig)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Apply LDA on the training set
    X_train_lda = lda.fit_transform(X_train, y_train)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_lda, y_train)

    # Apply LDA on the test set
    X_test_lda = lda.transform(X_test)

    # Calculate classification probabilities for the test set
    probs = model.predict_proba(X_test_lda)
    # Get the probability associated with class 1
    preds = probs[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # Calculate predictions on the test set
    y_pred = model.predict(X_test_lda)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display confusion matrix
    st.write("Matricea de confuzie:")
    st.write(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

    st.write("**3) t-SNE (t-distributed Stochastic Neighbor Embedding)**")

    with st.expander("Descrierea algoritmului"):
        st.write(
            "- Este un algoritm de reducere a dimensiunii folosit pentru vizualizarea datelor de mare dimensiune în două sau trei dimensiuni.")

    # Load data
    df = pd.read_csv("breast-cancer.csv")

    # Separate features and target
    X = df.drop(columns=["diagnosis"])
    Y = df["diagnosis"]

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Convert the results to a DataFrame
    X_tsne_df = pd.DataFrame(data=X_tsne, columns=['Componenta t-SNE 1', 'Componenta t-SNE 2'])

    # Plot t-SNE
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_tsne_df['Componenta t-SNE 1'], X_tsne_df['Componenta t-SNE 2'], c=Y, cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Diagnostice")
    ax.add_artist(legend)
    plt.xlabel('Componenta t-SNE 1')
    plt.ylabel('Componenta t-SNE 2')
    plt.title('Proiectarea t-SNE a datelor')
    st.pyplot(fig)


choose_MecanismComparatie = st.sidebar.button("Comparația Algoritmilor")
if choose_MecanismComparatie:
    st.header("Comparația Algoritmilor de Reducere a Dimensionalității")

    # Pregătirea datelor
    df = get_data()
    X = df.drop(columns=["diagnosis"])
    Y = df["diagnosis"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


    # Evaluare PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_accuracy = evaluate_model(LogisticRegression(), X_train_pca, X_test_pca, y_train, y_test)

    # Evaluare LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    lda_accuracy = evaluate_model(LogisticRegression(), X_train_lda, X_test_lda, y_train, y_test)

    # Evaluare t-SNE (folosit doar pentru vizualizare, nu pentru clasificare)
    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train)
    X_test_tsne = tsne.fit_transform(X_test)
    # Antrenarea unui model pe datele reduse cu t-SNE nu este recomandată deoarece t-SNE este destinat doar vizualizării.

    # Afisarea rezultatelor
    st.subheader("Comparația Algoritmilor de Reducere a Dimensionalității")

    st.write("**Acuratețea medie obținută:**")
    st.write(f"- PCA: {pca_accuracy * 100:.2f}%")
    st.write(f"- LDA: {lda_accuracy * 100:.2f}%")

    st.write(
        "**Notă:** t-SNE este utilizat în principal pentru vizualizare și nu pentru antrenarea modelelor predictive, așa că nu este inclus în comparația de acuratețe.")

    # Vizualizare PCA
    st.write("**Vizualizare PCA:**")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Diagnostice")
    ax.add_artist(legend)
    plt.xlabel('Componenta Principala 1')
    plt.ylabel('Componenta Principala 2')
    plt.title('Proiectarea PCA a datelor')
    st.pyplot(fig)

    # Vizualizare LDA
    st.write("**Vizualizare LDA:**")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_train_lda, [0] * len(X_train_lda), c=y_train, cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Diagnostice")
    ax.add_artist(legend)
    plt.xlabel('Componenta LDA 1')
    plt.ylabel('Valoare')
    plt.title('Proiectarea LDA a datelor')
    st.pyplot(fig)

    # Vizualizare t-SNE
    st.write("**Vizualizare t-SNE:**")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Diagnostice")
    ax.add_artist(legend)
    plt.xlabel('Componenta t-SNE 1')
    plt.ylabel('Componenta t-SNE 2')
    plt.title('Proiectarea t-SNE a datelor')
    st.pyplot(fig)

if (btn_afis_general or ((not choose_MecanismAtentie) and (not choose_tabel) and (not choose_MecanismComparatie))):
    from PIL import Image
    image_path = 'img/Capture.JPG'
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

st.sidebar.write('')
st.sidebar.write('Developer: **Andreea-Tabita Oprea**')
st.sidebar.write('Prof.: **Conf. Dr. Habil. Darian M. Onchiș**')
st.sidebar.write(
    "Universitatea de Vest Timisoara - Facultatea de Matematica și Informatica" + '\n')
st.sidebar.image('img/Logo-emblema-UVT-14.png', width=55)
st.sidebar.write('2023-2024')
st.set_option('deprecation.showPyplotGlobalUse', False)
