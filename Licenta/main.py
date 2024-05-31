from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,  roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

apptitle = 'BreastCancerAnalysis'
st.set_page_config(page_title=apptitle, page_icon=":bar_chart:")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


# fișierul CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

def get_data():
    return pd.read_csv('https://raw.githubusercontent.com/AA11nta/try/main/BreastCancerProject/breast-cancer.csv', header=0)

def new_alg(df):

    # Define features and target
    features = data.drop(columns=['diagnosis'])
    target = data['diagnosis']

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # Assuming 3 classes for the output
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    # Predict probabilities
    y_pred_probs = model.predict(X_test)

    # Convert probabilities to class labels
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC Curve and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Assuming the target has three classes, i.e., 0, 1, 2
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_probs[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Plot Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted labels')
    ax_cm.set_ylabel('True labels')
    ax_cm.set_title('Confusion Matrix')

    # Plot ROC curve
    fig_roc, ax_roc = plt.subplots()
    for i in range(3):
        ax_roc.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (area = {roc_auc[i]:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc="lower right")


    # Display confusion matrix in Streamlit
    st.write("### Matricea de confuzie")
    st.pyplot(fig_cm)

    # Display ROC curve in Streamlit
    st.write("### Curba ROC")
    st.pyplot(fig_roc)

    st.write("### Acuratețea")
    st.write(f"Rezultat: {accuracy:.2f}")
    st.line_chart(history.history['accuracy'])
    st.line_chart(history.history['val_accuracy'])

data = get_data()

st.sidebar.subheader("Prezentare generală a Aplicației")

btn_Introducere = st.sidebar.button("Introducere")
if btn_Introducere:
    st.title("Analiza Seturilor de Date pentru Cancerul Mamar: Abordări Avansate cu Inteligență Artificială")
    st.write(
        "În era digitală modernă, inteligența artificială (IA) joacă un rol crucial în îmbunătățirea diagnosticului medical și a tratamentelor personalizate. Una dintre cele mai provocatoare domenii de aplicare a acestor tehnologii este în diagnosticul și predicția cancerului mamar, o boală care afectează anual milioane de femei la nivel global. Proiectul de față își propune să dezvolte și să implementeze un model de rețea neuronală capabil să analizeze și să clasifice datele despre cancerul mamar, utilizând seturi de date publice și metode avansate de învățare automată."
        "\n\n"
        "Aplicația construită pentru acest proiect folosește un algoritm de rețea neuronală pentru a prelucra datele și a oferi predicții precise. Prin standardizarea caracteristicilor și utilizarea unor straturi multiple de rețea neuronală, modelul nostru poate învăța modele complexe și relații între datele de intrare și diagnosticul de ieșire. Pe lângă modelul principal, aplicația permite încărcarea și antrenarea altor seturi de date, oferind flexibilitate și extensibilitate în utilizare."
        "\n\n"
        "Acest proiect nu doar că demonstrează potențialul rețelelor neuronale în domeniul medical, dar și accentuează importanța prelucrării și analizei corecte a datelor pentru a obține rezultate fiabile și aplicabile în practica clinică. În continuare, vom detalia metodologia utilizată, structura modelului propus, rezultatele obținute și perspectivele de viitor ale acestei tehnologii în diagnosticul cancerului mamar."
        "\n"
        "\n"
    )

choose_Tabel = st.sidebar.button("Setul de date")
if (choose_Tabel):
    st.subheader("Setul de date")
    st.write(
        "Setul de date Breast Cancer de pe Kaggle, furnizat de Yasser Hesham, include informații despre diverse caracteristici ale tumorilor mamare."
        "\n\n"
        "Setul de date este utilizat pentru clasificare binară pentru a prezice dacă o tumoră este malignă sau benignă. Caracteristicile includ atribute precum raza, textura, perimetrul, aria, netezimea și altele.")
    df = get_data()
    # df = df.drop(columns="No")
    st.dataframe(df, height=450, hide_index=True)

    st.write("\n"
             "Setul de date conține informații despre 569 cazuri de cancer mamar. Fiecare caz este descris prin 30 de caracteristici diferite, care analizează trăsăturile nucleilor celulari prezenți în imaginea digitalizată a unei aspirate fine cu ac (FNA) a unei mase mamare. ")
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x='diagnosis', data=df, palette='hls')
    st.pyplot(fig)

    for col in df.columns:
        df.loc[(df["diagnosis"] == 0) & (df[col].isnull()), col] = df[df["diagnosis"] == 0][col].median()
        df.loc[(df["diagnosis"] == 1) & (df[col].isnull()), col] = df[df["diagnosis"] == 1][col].median()
    st.write(
        "<div style='text-align:justify;font-size: 16px;'>Mai jos puteți vizualiza histograma cu valorile pentru fiecare dintre componentele care influențează diagnosticul."
        "\n"
        "Cu cât punctele sunt mai răspândite, cu atât valorile sunt mai diverse. Locurile în care sunt strâns legate indică faptul că valorile respective sunt apropiate, reprezentând o majoritate."
        "<li style='color: #7e3e4f'>diagnosis: 0 - Persoane care au tumoare maligna</li>"
        "<li style='color: #3e517e'>diagnosis: 1 - Persoane care au tumoare beligna</li></div>",
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


choose_Algoritmi = st.sidebar.button("Algoritmi existenti")
if choose_Algoritmi:
    st.title("1. PCA (Principal Component Analysis)")

    with st.expander("Detalii algoritm:"):
        st.write("- **PCA (Principal Component Analysis)** este o tehnică de reducere a dimensiunii datelor folosită în analiza statistică și în învățarea automată. Scopul principal al PCA este de a transforma un set mare de variabile corelate într-un set mai mic de variabile necorelate, numite componente principale.")
        st.write("- Folosirea PCA (Principal Component Analysis) oferă multiple **avantaje** în analiza datelor și învățarea automată. În primul rând, reduce dimensiunea datelor păstrând variabilitatea esențială, ceea ce simplifică analiza și vizualizarea și reduce timpul și resursele computaționale necesare. De asemenea, elimină redundanța informațiilor prin combinarea variabilelor corelate în componente principale necorelate, rezultând modele mai eficiente și mai rapide. PCA îmbunătățește performanța algoritmilor de învățare automată, reducând riscul de overfitting și crescând acuratețea modelelor. Facilitează vizualizarea datelor multidimensionale în două sau trei dimensiuni, ajutând la identificarea tiparelor, grupurilor sau outlier-ilor. Permite compresia datelor, economisind spațiu de stocare și păstrând informațiile relevante. PCA este utilizată ca un pas de preprocesare pentru a îmbunătăți calitatea datelor și a facilita antrenarea modelelor și ajută la reducerea zgomotului, extrăgând componentele principale care reprezintă cele mai semnificative variații, îmbunătățind claritatea și relevanța datelor. Aceste avantaje fac din PCA un instrument valoros în analiza datelor și dezvoltarea modelelor predictive, mai ales pentru seturile de date mari și complexe.")

    # Load dataset
    data = pd.read_csv('breast-cancer.csv')

    # Preprocessing
    X = data.drop(columns=['No', 'diagnosis'])
    y = data['diagnosis']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA Implementation
    pca = PCA(n_components=2)  # Reduce to 2 components for visualization
    X_pca = pca.fit_transform(X_scaled)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # Train Logistic Regression Classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Compute confusion matrix and accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    st.write("### Aplicarea algoritmului pe setul de date dat")
    col1, col2 = st.columns(2, gap='large')

    with col1:
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Diagnosis")
        ax.add_artist(legend1)
        st.pyplot(fig)

        st.write("### Matricea de confuzie")
        # st.write(conf_matrix)
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt)


    with col2:
        st.write("### Curba ROC")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

        st.write("### Acuratețea")
        st.write(f"Rezultat: {accuracy:.2f}")



    st.title("LDA (Linear Discriminant Analysis)")

    with st.expander("Detalii algoritm:"):
        st.write("- Linear Discriminant Analysis (LDA) este o tehnică de învățare supravegheată utilizată atât pentru reducerea dimensiunii, cât și pentru clasificare. Scopul principal al LDA este de a găsi o proiecție liniară a datelor care maximizează separabilitatea între clase. Aceasta se realizează prin calcularea mediilor și matricelor de dispersie pentru fiecare clasă, urmată de identificarea vectorilor proprii care maximizează raportul dintre dispersia între clase și dispersia în interiorul clasei. Rezultatul este un set de axe pe care datele sunt proiectate, reducând dimensiunea datelor și menținând separabilitatea maximă între clase.")
        st.write("- LDA este avantajos pentru reducerea dimensiunii, păstrând caracteristicile relevante pentru clasificare și îmbunătățind performanța clasificatorilor. Totuși, presupune că datele sunt normal distribuite și că matricile de covarianță ale claselor sunt egale, ipoteze care pot să nu fie întotdeauna îndeplinite. În ciuda acestor limitări, LDA este o metodă populară datorită eficienței și simplității sale, fiind aplicată cu succes în diverse domenii, inclusiv recunoașterea facială și bioinformatică.")


    # Separate features and target
    df = get_data()
    X = df.drop(columns=["diagnosis"])
    Y = df["diagnosis"]

    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X, Y)

    # Convert the results to a DataFrame
    X_lda_df = pd.DataFrame(data=X_lda, columns=['Componenta LDA 1'])

    st.write("### Aplicarea algoritmului pe setul de date dat")
    col1, col2 = st.columns(2, gap='large')
    with col1:
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

        # Calculate predictions on the test set
        y_pred = model.predict(X_test_lda)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Display confusion matrix
        st.write("### Matricea de confuzie")
        # st.write(conf_matrix)

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

    with col2:

        st.write("### Curba ROC")
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

        accuracy = accuracy_score(y_test, y_pred)
        st.write("### Acuratețea:")
        st.write(f"Rezultat: {accuracy:.2f}")

    st.title("t-SNE (t-distributed Stochastic Neighbor Embedding)")

    with st.expander("Detalii algoritm:"):
        st.write(
            "- t-SNE (t-distributed Stochastic Neighbor Embedding) este o tehnică de reducere a dimensiunii utilizată în principal pentru vizualizarea datelor de înaltă dimensiune. Dezvoltată de Laurens van der Maaten și Geoffrey Hinton, t-SNE mapează datele de înaltă dimensiune într-un spațiu cu două sau trei dimensiuni, păstrând relațiile de proximitate între punctele de date. Algoritmul folosește o distribuție t-student pentru a modela distanțele mari, ceea ce ajută la gestionarea mai eficientă a aglomerărilor și la evitarea suprapunerii punctelor în reprezentările bidimensionale sau tridimensionale.")

    # Load data
    df = pd.read_csv("breast-cancer.csv")

    # Preprocessing
    X = data.drop(columns=['No', 'diagnosis'])
    y = data['diagnosis']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE Implementation
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_tsne, y, test_size=0.3, random_state=42)

    # Train Logistic Regression Classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Compute confusion matrix and accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    st.write("### Aplicarea algoritmului pe setul de date dat")
    col1, col2 = st.columns(2, gap='large')
    with col1:
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Diagnosis")
        ax.add_artist(legend1)
        st.pyplot(fig)

        st.write("### Matricea de confuzie")
        # st.write(conf_matrix)
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

    with col2:
        st.write("### Curba ROC")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

        st.write("### Acuratețea")
        st.write(f"Rezultat: {accuracy:.2f}")

choose_MecanismReteaNeuronala = st.sidebar.button("Model de Rețea Neuronală")
if choose_MecanismReteaNeuronala:
    st.header("Crearea și antrenarea modelului de rețea neuronală")
    with st.expander("Descrierea algoritmului"):
        st.write("- Algoritmul este un proces de învățare automată destinat clasificării datelor utilizând o rețea neuronală artificială. Începe prin preprocesarea datelor, unde caracteristicile sunt extrase și standardizate pentru a asigura o scalare consistentă. Apoi, datele sunt împărțite în seturi de antrenament și testare pentru a permite evaluarea performanței modelului.")
        st.write("- Rețeaua neuronală este definită utilizând biblioteca TensorFlow/Keras, constând dintr-un model secvențial cu mai multe straturi dense (fully connected). Fiecare strat folosește funcția de activare ReLU pentru a introduce non-linearități, în timp ce stratul final folosește funcția de activare softmax pentru a produce probabilitățile de clasificare pentru cele trei clase presupuse. Modelul este compilat folosind optimizatorul Adam și funcția de pierdere sparse_categorical_crossentropy, potrivită pentru problemele de clasificare multi-clasă. Modelul este antrenat pe datele de antrenament, iar performanța sa este evaluată pe datele de testare.")
        st.write("- După antrenare, modelul este evaluat folosind diverse metrici, inclusiv acuratețea și matricea de confuzie, pentru a evalua performanța clasificării. De asemenea, sunt generate curbele ROC (Receiver Operating Characteristic) și valorile AUC (area under the curve) pentru fiecare clasă, oferind o evaluare detaliată a capacității modelului de a diferenția între clase. ")

    # Load the dataset
    df = get_data()
    new_alg(df)

choose_ModelNou = st.sidebar.checkbox("Antrenează un alt set de date")
if choose_ModelNou:
    st.header("Set de date încărcat de utilizator")
    uploaded_file = st.file_uploader("*Alege un fisier", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("- Fisiserul încărcat:")
        st.write(df)
        new_alg(df)

if (( not btn_Introducere) and (not choose_Algoritmi) and (not choose_Tabel) and (not choose_ModelNou) and (not choose_MecanismReteaNeuronala)):
    st.title("Analiza Seturilor de Date pentru Cancerul Mamar: Abordări Avansate cu Inteligență Artificială")


st.sidebar.write('')
st.sidebar.write('Student: **Andreea-Tabita Oprea**')
st.sidebar.write('Prof.: **Conf. Dr. Habil. Darian M. Onchiș**')
st.sidebar.write(
    "Universitatea de Vest Timisoara - Facultatea de Matematica și Informatica" )

st.sidebar.write('2023-2024')

st.sidebar.image('img/Logo-emblema-UVT-14.png', width=100)
st.set_option('deprecation.showPyplotGlobalUse', False)
