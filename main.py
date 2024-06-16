import math
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from scalers.invertedminmaxscaler import InvertedMinMaxScaler
from scalers.thresholdminmaxscaler import ThresholdMinMaxScaler
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import seaborn as sns
# st.set_page_config(layout="wide")
plt.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white'
})

'''
# University Ranking System
'''


def display_university_data_editor(excel_file) -> pd.DataFrame:
    df = pd.read_excel(excel_file, sheet_name='University')
    df = st.data_editor(df)
    df.set_index('University', inplace=True)

    return df


def get_feature_df(excel_file) -> pd.DataFrame:
    df = pd.read_excel(excel_file, sheet_name='Feature')
    df.set_index('Feature', inplace=True)

    return df


def display_feature_description_table(feature_df):
    st.table(feature_df['Description'])


def validate_data(university_df, feature_df):
    university_df_features = list(university_df.columns)
    feature_df_features = list(feature_df.index)

    assert university_df_features == feature_df_features


def get_scaler(scaler: str, **kwargs):
    if scaler == 'MinMax':
        return MinMaxScaler()
    elif scaler == 'MinMax_Inverted':
        return InvertedMinMaxScaler()
    elif scaler == 'MinMax_Threshold':
        return ThresholdMinMaxScaler(kwargs['Threshold'])
    else:
        raise ValueError(f'Unknown Scaler: {scaler}')


def get_preprocessor(feature_df):
    transformers = []
    for _, row in feature_df.iterrows():
        # Scaler Name
        scaler_name = f"{row.name}_Scaler"

        # Scaler Implementation
        scaler_kwargs = {}
        if not math.isnan(row['Threshold']):
            scaler_kwargs['Threshold'] = row['Threshold']
        scaler = get_scaler(row['Scaler'], **scaler_kwargs)

        # Scaler Column
        column = row.name

        # Add to transformers
        transformers.append((scaler_name, scaler, [column]))

    normalization_preprocessor = ColumnTransformer(
        remainder='passthrough',
        transformers=transformers
    )

    return normalization_preprocessor


def get_normalized_university_df(university_df, normalization_preprocessor):
    university_df_features = list(university_df.columns)

    university_df_data = university_df[university_df_features]
    normalized_university_df = normalization_preprocessor.fit_transform(university_df_data)
    normalized_university_df = pd.DataFrame(normalized_university_df)
    normalized_university_df.columns = university_df_features

    normalized_university_df.insert(0, 'University', university_df.index)
    normalized_university_df.set_index('University', inplace=True)

    return normalized_university_df


def get_imputed_university_df(university_df):
    university_df_features = list(university_df.columns)

    mean_imputer = SimpleImputer()
    imputed_normalized_university_df = mean_imputer.fit_transform(university_df)
    imputed_normalized_university_df = pd.DataFrame(imputed_normalized_university_df)
    imputed_normalized_university_df.columns = university_df_features

    imputed_normalized_university_df.insert(0, 'University', university_df.index)
    imputed_normalized_university_df.set_index('University', inplace=True)

    return imputed_normalized_university_df


def get_university_score_df(university_df):
    weighted_score_df = university_df.mean(axis=1).sort_values()

    return weighted_score_df


def get_weighted_university_df(normalized_university_df, feature_df):
    weighted_university_df = normalized_university_df.copy()

    for _, row in feature_df.iterrows():
        weighted_university_df[row.name] *= (row['Weight'] / 100)

    return weighted_university_df


def display_heatmap(university_df):
    fig, ax = plt.subplots(facecolor='#0f1018')

    sns.heatmap(university_df, annot=True, cmap='RdYlGn', cbar=True, linewidths=0.5, linecolor='black')
    ax.set_ylabel('')

    st.pyplot(fig, use_container_width=True)


def display_radar_charts(imputed_normalized_university_df):
    university_df_features = list(imputed_normalized_university_df.columns)

    fig = go.Figure()

    for _, row in imputed_normalized_university_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=list(row),
            theta=university_df_features,
            fill='toself',
            name=row.name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False
            )),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def display_barchart(university_score_df):
    fig, ax = plt.subplots(facecolor='#0f1018')

    ax.barh(university_score_df.index, university_score_df)

    st.pyplot(fig, use_container_width=True)


def display_weight_sliders(feature_df):
    for idx, row in feature_df.iterrows():
        feature_df.at[idx, 'Weight'] = st.slider(row.name, 0, 100, row['Weight'])

    return feature_df


def main():
    st.header('Upload :blue[Excel] File', divider='gray')
    excel_file = st.file_uploader("Upload the Excel file", type=('xlsx', 'xls'))
    if excel_file is not None:
        st.header(':blue[Feature] Explanation', divider='gray')
        feature_df = get_feature_df(excel_file)
        display_feature_description_table(feature_df)

        st.header(':blue[University] Data', divider='gray')
        university_df = display_university_data_editor(excel_file)
        validate_data(university_df, feature_df)

        st.subheader('Normalized Data Heatmap', divider='gray')
        normalization_preprocessor = get_preprocessor(feature_df)
        normalized_university_df = get_normalized_university_df(university_df, normalization_preprocessor)
        display_heatmap(normalized_university_df)

        st.subheader('Normalized Features Radar Chart', divider='gray')
        st.text('(If there are missing values the software imputes the values based on the average)')
        imputed_normalized_university_df = get_imputed_university_df(normalized_university_df)
        display_radar_charts(imputed_normalized_university_df)

        st.header('Apply :blue[Weights] to University Data', divider='gray')
        feature_df = display_weight_sliders(feature_df)

        st.subheader('Weighted Data Heatmap', divider='gray')
        weighted_university_df = get_weighted_university_df(normalized_university_df, feature_df)
        display_heatmap(weighted_university_df)

        st.subheader('Score based on Average of Weighted Data', divider='gray')
        st.text('(If there are missing values the software imputes the values based on the average)')
        imputed_weighted_university_df = get_imputed_university_df(weighted_university_df)
        weighted_university_score_df = get_university_score_df(imputed_weighted_university_df)
        display_barchart(weighted_university_score_df)


if __name__ == '__main__':
    main()
