import numpy as np
import streamlit as st
import pickle

svm = pickle.load(open('SVC_model.pkl','rb'))
grid = pickle.load(open('grid.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))
log_model = pickle.load(open('log_model.pkl','rb'))

def classify(species):
    output = np.array_str(species)
    if output == "['setosa']":
        return 'Setosa'
    elif output == "['versicolor']":
        return 'Versicolor'
    elif output == "['virginica']":
        return 'Virginica'
    else:
        return type(output)


def main():
    st.title("Iris Classifier")
    html_temp = """
    <br>
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;"><b>Iris Classification</b></h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['SVM', 'GridSearch', 'RandomForest', 'Logistic Regression']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    sl=st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length', 0.0, 10.0)
    pw=st.slider('Select Petal Width', 0.0, 10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button('Classify'):
        if option=='SVM':
            st.success(classify(svm.predict(inputs)))
        elif option=='GridSearch':
            st.success(classify(grid.predict(inputs)))
        elif option=='RandomForest':
            st.success(classify(rf.predict(inputs)))
        else:
            st.success(classify(log_model.predict(inputs)))
            
    st.markdown(
    """<a style='display: block; text-align: center;' href="https://github.com/Manish-Sharma-1810/IrisClassifier-App" target="_blank">Source code</a>
       <a style='display: block; text-align: center;' href="https://github.com/Manish-Sharma-1810/IrisClassifier-App/blob/main/iris_models.ipynb" target="_blank">Jupyter File</a>
       <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/manish-kumar-1810?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BSvgfK3HQSWC506NySLl1PQ%3D%3D" target="_blank">LinkedIn</a>
    """, unsafe_allow_html=True)


if __name__=='__main__':
    main()
