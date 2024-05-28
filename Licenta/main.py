from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
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

choose_MecanismAtentie = st.sidebar.button("Algoritmul PCA")
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

    st.write("**2) The attention mechanism**")

    with st.expander("The description of the mechanism"):
        st.write("- It is a technique used to enable the model to focus on the relevant parts of the input "
                 "data during the learning process")
        st.write("- The primary purpose of the mechanism is to contribute to improving performance and gaining a deeper"
                 " understanding of the input data.")

    with st.expander("The description of the algorithm used."):
        st.write(
            "- The developed prediction model consists of two types of layers, namely: **Dense și BatchNormalization**")
        st.write(
            "- The **Dense** layer is used in neural networks, where each neuron in the current layer connects to all "
            "neurons in the next layer. It is a fully connected layer, where each neuron receives all input values from "
            "the previous layer and produces an output value.")
        st.write(
            "- The **BatchNormalization** layer is a layer used in deep neural networks to normalize activations"
            " between layers during the training process. It was introduced to help speed up training, "
            "reduce overfitting, and improve the model's generalization.")
    st.write("- After training the model, the following results were achieved:")

    # df_nou = df[positive_features].copy()
    # X_train_nou, X_test_nou, y_train_nou, y_test_nou = train_test_split(X_nou, y, test_size=0.2, random_state=0)
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(128, activation="relu", input_shape=([df_nou.shape[1] - 1])),
    #     tf.keras.layers.BatchNormalization(axis=-1),
    #     tf.keras.layers.Dense(64, activation="relu"),
    #     tf.keras.layers.BatchNormalization(axis=-1),
    #     tf.keras.layers.Dense(8, activation="relu"),
    #     tf.keras.layers.Dense(1, activation="sigmoid"),
    # ])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # model.fit(X_train_nou, y_train_nou, epochs=10, batch_size=32, validation_data=(X_test_nou, y_test_nou))

    # y_pred_nou = model.predict(X_test_nou)
    # y_pred_nou = y_pred_nou > 0.45
    #
    # np.set_printoptions()
    #
    # cm = confusion_matrix(y_test_nou, y_pred_nou)
    # ac = accuracy_score(y_test_nou, y_pred_nou)
    #
    # col1, col2 = st.columns(2, gap='large')
    #
    # with col1:
    #     st.write("a) The **confusion matrix** provides a tabular representation of classification results, "
    #              "comparing the model's predictions with the actual values of class labels.")
    #
    #     disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    #     disp.plot()
    #     plt.title('Confusion matrix')
    #     plt.show()
    #     st.pyplot()
    #
    #     st.write(
    #         "1) True Positive (TP): Represents the number of instances for which the model correctly predicted "
    #         "the positive class.")
    #     st.write(
    #         "2) True Negative (TN): Represents the number of instances for which the model correctly predicted"
    #         " the negative class")
    #     st.write(
    #         "3) False Positive (FP): Represents the number of instances for which the model incorrectly predicted "
    #         "the positive class (predicted it belongs to the positive class when, in reality, it doesn't)")
    #     st.write(
    #         "4) False Negative (FN): Represents the number of instances for which the model incorrectly \
    #         predicted the negative class (predicted it belongs to the negative class when, in reality,"
    #         " it belongs to the positive class).")
    # with col2:
    #     st.write(
    #         "b) The ROC curve (Receiver Operating Characteristic) is a method used to evaluate the performance of a "
    #         "binary classification model. It provides a graphical representation of the trade-off between the"
    #         " true positive rate and the false positive rate as the model's decision thresholds are varied.")
    #     fpr, tpr, thresholds = metrics.roc_curve(y_test_nou, y_pred_nou)
    #     plt.axis('scaled')
    #     plt.xlim([0.1, 0.9])
    #     plt.ylim([0.1, 0.9])
    #     plt.title('Curba ROC')
    #     plt.plot(fpr, tpr, 'b')
    #     plt.fill_between(fpr, tpr, facecolor='lightblue', alpha=0.5)
    #     plt.ylabel('True Positive Rate')
    #     plt.xlabel('False Positive Rate')
    #     plt.show()
    #     st.pyplot()
    #
    #     st.write(
    #         "An ideal ROC curve approaches the upper-left corner of the graph, indicating a model with a high "
    #         "true positive rate (TPR) and a low false positive rate (FPR).")
    #     st.write(
    #         "Greater AUC (Area Under the Curve) indicates better model performance.")
    #     st.write("- The ROC curve provides a way to assess the trade-off between true positive detection rates and "
    #              "false positive detection rates, enabling the selection of the optimal decision"
    #              " threshold for classification.")

st.sidebar.write('')
st.sidebar.write('Developer: **Andreea-Tabita Oprea**')
st.sidebar.write('Prof.: **Conf. Dr. Habil. Darian M. Onchiș**')
st.sidebar.write(
    "Universitatea de Vest Timisoara - Facultatea de Matematica și Informatica" + '\n')
st.sidebar.image('img/Logo-emblema-UVT-14.png', width=55)
st.sidebar.write('2023-2024')
st.set_option('deprecation.showPyplotGlobalUse', False)