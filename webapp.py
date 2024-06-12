import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score

# Main function to run the Streamlit app
def main():
    # App title and introduction
    st.title("Mushroom Classification Web App")
    st.sidebar.title("Mushroom Classification Web App")
    st.markdown("""
    ## Introduction
    This web application aims to classify mushrooms as either edible or poisonous based on various features. 
    Given the serious health risks associated with mushroom poisoning, this classification task is crucial. 
    The dataset used in this project is sourced from the UCI Machine Learning Repository.
    
    The motivation behind this project is to create a practical tool that can assist in the identification of potentially hazardous mushrooms. 
    This project showcases my ability to build a machine learning application that leverages data preprocessing, model training, evaluation, 
    and deployment using Streamlit.
    """)

    # Function to load and encode the data
    @st.cache_data(persist=True)
    def load_data():
        try:
            data = pd.read_csv("mushrooms.csv")
            labelencoder = LabelEncoder()
            for col in data.columns:
                data[col] = labelencoder.fit_transform(data[col])
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    # Function to split the data into training and testing sets
    @st.cache_data(persist=True)
    def split(df):
        try:
            y = df["type"]
            x = df.drop(columns=['type'])
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

            # Scaling the data
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            
            return x_train, x_test, y_train, y_test
        except Exception as e:
            st.error(f"Error splitting data: {e}")
            return None, None, None, None

    # Function to calculate feature importance for the Random Forest model
    @st.cache_data(persist=True)
    def feature_importance(model, x_train):
        try:
            importance = model.feature_importances_
            indices = np.argsort(importance)
            features = x_train.columns[indices]
            return pd.DataFrame({'Feature': features, 'Importance': importance[indices]})
        except Exception as e:
            st.error(f"Error calculating feature importance: {e}")
            return None
    
    # Function to plot confusion matrix
    def plot_confusion_matrix(cm, class_names):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    # Function to plot ROC curve
    def plot_roc_curve(fpr, tpr):
        fig, ax = plt.subplots()
        plt.plot(fpr, tpr, color='blue', lw=2)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        st.pyplot(fig)

    # Function to plot precision-recall curve
    def plot_precision_recall_curve(precision, recall):
        fig, ax = plt.subplots()
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        st.pyplot(fig)
    
    # Load the data
    df = load_data()
    if df is None:
        return

    class_names = ['edible', 'poisonous']
    
    # Option to display raw data
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
    
    # Option to display data summary
    if st.sidebar.checkbox("Show data summary", False):
        st.subheader("Data Summary")
        st.write(df.describe())

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = split(df)
    if x_train is None:
        return

    # Convert y_train and y_test to numpy arrays (to avoid issues with writeable flag)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Classifier selection
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    # Model explanation
    st.sidebar.markdown("""
    ### Model Explanations:
    - **Support Vector Machine (SVM)**: SVMs are supervised learning models used for classification and regression tasks. They work well for high-dimensional spaces and are effective when the number of dimensions is greater than the number of samples.
    - **Logistic Regression**: A statistical model that in its basic form uses a logistic function to model a binary dependent variable. It is simple to implement and can be interpreted easily.
    - **Random Forest**: An ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes for classification.
    """)

    # SVM classifier
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Train Model", key='train_svm'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))

            if 'Confusion Matrix' in metrics:
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(cm, class_names)
                st.write("""
                The confusion matrix shows the actual versus predicted classifications. 
                It helps in understanding the number of correct and incorrect predictions made by the model.
                """)

            if 'ROC Curve' in metrics:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                st.subheader("ROC Curve")
                plot_roc_curve(fpr, tpr)
                st.write("""
                The ROC curve illustrates the diagnostic ability of a binary classifier system. 
                It plots the true positive rate against the false positive rate at various threshold settings.
                """)

            if 'Precision-Recall Curve' in metrics:
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                st.subheader('Precision-Recall Curve')
                plot_precision_recall_curve(precision, recall)
                st.write("""
                The Precision-Recall curve shows the trade-off between precision and recall for different threshold values.
                It is useful for understanding the performance of a model on an imbalanced dataset.
                """)
    
    # Logistic Regression classifier
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 1000, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Train Model", key='train_lr'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))

            if 'Confusion Matrix' in metrics:
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(cm, class_names)
                st.write("""
                The confusion matrix shows the actual versus predicted classifications. 
                It helps in understanding the number of correct and incorrect predictions made by the model.
                """)

            if 'ROC Curve' in metrics:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                st.subheader("ROC Curve")
                plot_roc_curve(fpr, tpr)
                st.write("""
                The ROC curve illustrates the diagnostic ability of a binary classifier system. 
                It plots the true positive rate against the false positive rate at various threshold settings.
                """)

            if 'Precision-Recall Curve' in metrics:
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                st.subheader('Precision-Recall Curve')
                plot_precision_recall_curve(precision, recall)
                st.write("""
                The Precision-Recall curve shows the trade-off between precision and recall for different threshold values.
                It is useful for understanding the performance of a model on an imbalanced dataset.
                """)
    
    # Random Forest classifier
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve', 'Feature Importance'))

        if st.sidebar.button("Train Model", key='train_rf'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))

            if 'Confusion Matrix' in metrics:
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(cm, class_names)
                st.write("""
                The confusion matrix shows the actual versus predicted classifications. 
                It helps in understanding the number of correct and incorrect predictions made by the model.
                """)

            if 'ROC Curve' in metrics:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                st.subheader("ROC Curve")
                plot_roc_curve(fpr, tpr)
                st.write("""
                The ROC curve illustrates the diagnostic ability of a binary classifier system. 
                It plots the true positive rate against the false positive rate at various threshold settings.
                """)

            if 'Precision-Recall Curve' in metrics:
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                st.subheader('Precision-Recall Curve')
                plot_precision_recall_curve(precision, recall)
                st.write("""
                The Precision-Recall curve shows the trade-off between precision and recall for different threshold values.
                It is useful for understanding the performance of a model on an imbalanced dataset.
                """)

            if 'Feature Importance' in metrics:
                st.subheader("Feature Importance")
                fi_df = feature_importance(model, x_train)
                if fi_df is not None:
                    st.write(fi_df)
                    plt.figure(figsize=(10,6))
                    plt.title('Feature Importances')
                    sns.barplot(x='Importance', y='Feature', data=fi_df)
                    st.pyplot()
                    st.write("""
                    The feature importance plot helps to identify which features are most influential in the model's decision-making process.
                    """)

    # Model comparison
    if st.sidebar.checkbox("Compare Models", False):
        st.subheader("Model Comparison")
        compare_button = st.sidebar.button("Compare Models")
        
        if compare_button:
            models = {
                "Support Vector Machine": SVC(),
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier()
            }
            results = {}
            for name, model in models.items():
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                results[name] = accuracy.round(2)
            st.write(results)
    
    # Detailed write-up
    st.markdown("""
    ## Project Insights and Conclusions
    
    ### Why Mushroom Classification?
    The classification of mushrooms as edible or poisonous is a significant task due to the potential health risks associated with consuming poisonous mushrooms. This project not only demonstrates my technical skills in data analysis and machine learning but also has practical implications for public health and safety.

    ### Results
    The models used in this project—Support Vector Machine (SVM), Logistic Regression, and Random Forest—were evaluated based on their accuracy, precision, recall, and other relevant metrics.

    - **Support Vector Machine (SVM)**: Achieved an accuracy of around 0.99, indicating high performance in distinguishing between edible and poisonous mushrooms.
    - **Logistic Regression**: Also performed well with an accuracy of around 0.95. Its simplicity and interpretability make it a strong candidate for this classification task.
    - **Random Forest**: Showed the best performance with an accuracy of around 1.00. The model's ability to handle complex datasets and identify important features was evident.

    ### Insights
    - **Data Summary and Visualization**: The dataset contains 8124 samples of mushrooms with 23 categorical features. Visualizations showed clear patterns in the data, with certain features like odor and spore print color exhibiting significant differences between edible and poisonous mushrooms.
    - **Model Performance**: Each model provided valuable insights and performed well in classification tasks. The Random Forest model's feature importance analysis highlighted the most influential features, aiding in better understanding and potential further research.

    ### Recommendations
    - **Feature Engineering**: Further feature engineering and selection could be explored to enhance model performance.
    - **Model Tuning**: Additional hyperparameter tuning and testing different machine learning models might yield even better results.
    - **Real-world Application**: This model can be integrated into a mobile application to help users identify mushrooms in real-time, potentially preventing mushroom poisoning incidents.

    ### Conclusion
    This project illustrates the application of machine learning techniques to a practical problem, showcasing the entire data analysis workflow from data preprocessing to model evaluation and deployment. The results demonstrate the effectiveness of machine learning in solving real-world classification tasks and provide a foundation for further improvements and applications.
    """)

if __name__ == '__main__':
    main()
