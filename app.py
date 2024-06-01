import streamlit as st
from bace import predict
import pandas as pd

st.set_page_config(layout="wide")

st.markdown("<center><h1>ADMET<small><br> Powerful </small></h1>", unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

col1,col2=st.columns(2)
with col1:
    smi = st.text_input("SMILES:",r"CC(C)C[n+]1ccc2c3ccccc3n3c(=O)ccc1c23")
#smi = st.file_uploader("Choose a file")
with col2:
	tar = st.selectbox(
    'Target:',
    ('HLM_1','HLM_2','HLM_3','RLM_1','RLM_2','RLM_3','UGTs_1','UGTs_2','UGTs_3','Pgp-inhibitor_1','Pgp-inhibitor_2','Pgp-inhibitor_3'))

    

#st.image('ha.jpg')


if st.button('Predict'):
    with st.spinner("Loading"):
        ans=predict(tar,smi)
        if ans == -1:
            st.error('Invalid SMILES!', icon="❌")
        else:
            #st.write(ans)
            
            #df = pd.DataFrame(ans, columns=['haha', 'hihi'])
            #st.dataframe(df, use_container_width=True)


            
            if ans>=0.5:
                flag='Positive'
                st.balloons()
            else:
                flag='Negative'
                st.snow()
            st.success(f"Possibility: {format('%.3f'%ans)}, {flag}", icon="🚦")
            