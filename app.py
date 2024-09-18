import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import joblib
import io

# Load your trained models
model_morgan2_1 = joblib.load('best_model_rf_knm2.pkl')
model_morgan2_2 = joblib.load('best_model_rf_pm2.pkl')
#model_morgan2_3 = joblib.load('best_model_lgb_tm2.pkl')
model_morgan2_4 = joblib.load('best_model_xgb_gm2.pkl')

#model_morgan3_1 = joblib.load('best_model_knn_trm3.pkl')
model_morgan3_2 = joblib.load('best_model_xgb_am3.pkl')
#model_morgan3_2 = joblib.load('xgb_model_acyl.pkl')
model_morgan3_3 = joblib.load('best_model_lgb_gtm3.pkl')
#model_morgan3_4 = joblib.load('lr_model.pkl')

# Function to generate Morgan fingerprints
def generate_morgan_fingerprint(smiles, radius):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius)
    return fp

# Streamlit app
st.title("Ligand Activity Prediction")

input_option = st.radio("Select input type:", ("Single SMILES string", "Upload SMILES file"))

smiles_list = []

if input_option == "Single SMILES string":
    smiles_input = st.text_input("Enter a ligand SMILES string:")
    if smiles_input:
        smiles_list = [smiles_input]

elif input_option == "Upload SMILES file":
    uploaded_file = st.file_uploader("Upload a file containing SMILES strings:", type=['txt', 'csv'])
    if uploaded_file is not None:
        # Read file and extract SMILES
        smiles_df = pd.read_csv(uploaded_file)
        smiles_list = smiles_df.iloc[:, 0].tolist()

if smiles_list:
    results = []
    for smiles in smiles_list:
        morgan2_fp = generate_morgan_fingerprint(smiles, radius=2)
        morgan3_fp = generate_morgan_fingerprint(smiles, radius=3)

        if morgan2_fp is not None and morgan3_fp is not None:
            morgan2_fp = [morgan2_fp]  # Convert to 2D array for prediction
            morgan3_fp = [morgan3_fp]

            # Predict with Morgan 2 models
            prediction_morgan2_1 = model_morgan2_1.predict(morgan2_fp)[0]
            prediction_morgan2_2 = model_morgan2_2.predict(morgan2_fp)[0]
            #prediction_morgan2_3 = model_morgan2_3.predict(morgan2_fp)[0]
            prediction_morgan2_4 = model_morgan2_4.predict(morgan2_fp)[0]

            # Predict with Morgan 3 models
            #prediction_morgan3_1 = model_morgan3_1.predict(morgan3_fp)[0]
            prediction_morgan3_2 = model_morgan3_2.predict(morgan3_fp)[0]
            prediction_morgan3_3 = model_morgan3_3.predict(morgan3_fp)[0]
            #prediction_morgan3_4 = model_morgan3_4.predict(morgan3_fp)[0]

            result = {
                'SMILES': smiles,
                'Kinase_Bioclass': prediction_morgan2_1,
                'Peptidase_Bioclass': prediction_morgan2_2,
                #'Topoisomerase_Bioclass': prediction_morgan2_3,
                'Glycosylase_Bioclass': prediction_morgan2_4,
               # 'Transaminase_Bioclass': prediction_morgan3_1,
                'Acyltransferase_Bioclass': prediction_morgan3_2,
                'Glycotranferase_Bioclass': prediction_morgan3_3,
                #'Kinesin-linked_Bioclass': prediction_morgan3_4,
            }
            results.append(result)
        else:
            st.error(f"Invalid SMILES string: {smiles}")

    if results:
        results_df = pd.DataFrame(results)
        st.write(results_df)

        # Allow the user to download the results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='ligand_activity_predictions.csv',
            mime='text/csv'
        )