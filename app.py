from flask import Flask, request, render_template, jsonify
import pandas as pd
import json
import dice_ml
from dice_ml import Dice
import pickle
import pandas as pd
from dice_ml import Dice
import dice_ml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Construir la ruta al archivo del modelo
model_path = os.path.join(os.getcwd(), "PRODUCTO DE DATOS", "models", "loaded_model.pkl")

# Verificar si el archivo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el archivo del modelo en la ruta: {model_path}")

# Cargar el modelo entrenado
with open(model_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)

print("Modelo cargado exitosamente.")

# Función para agrupar y transformar el DataFrame
def group_and_transform(df):
    # # Listas de columnas categóricas y numéricas esperadas
    categorical_columns = ['SEXO', 'ESTADO', 'ESTADO_CIVIL', 'TIPO_COLEGIO', 'CARRERA', 'PER_INGRESO', 'PER_MATRICULA', 'CURSO', 'GRUPO', 'Departamento_Procedencia', 'Provincia_Procedencia', 'Distrito_Procedencia', 'Departamento_Residencia', 'Provincia_Residencia', 'Distrito_Residencia']
    numeric_columns = ['SEM_ALUMNO', 'SEM_CURSADOS', 'CANT_RESERVAS', 'PTJE_INGRESO', 'COD_CURSO', 'CREDITOS', 'TIPO_CURSO', 'COD_PLAN', 'COD_GRUPO', 'NOTA', 'HRS_INASISTENCIA', 'HRS_CURSO', 'PRCTJE_INASISTENCIA', 'PONDERADO', 'CRED_GRADUACION', 'BECA_VIGENTE', 'NOTA_ENCUESTA_DOC', 'POBLACION', 'IDH', 'POR_POBREZA', 'POR_POBREZA_EXTREMA', 'EDAD']
    
    # Validar y corregir tipos de datos
    errors = {}
    for column in df.columns:
        if column in categorical_columns:
            # Verificar si la columna categórica contiene solo valores de texto
            if not pd.api.types.is_string_dtype(df[column]):
                try:
                    df[column] = df[column].astype(str)  # Convertir a texto
                except ValueError:
                    errors[column] = "Expected categorical (string) values but found incompatible data."
        elif column in numeric_columns:
            # Verificar si la columna numérica contiene solo valores numéricos
            if not pd.api.types.is_numeric_dtype(df[column]):
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except ValueError:
                    errors[column] = "Expected numeric values but found incompatible data."
    
    # Imputación de valores nulos: numéricos con media y categóricos con moda
    for column in df.columns:
        if df[column].dtype == 'object':  # Para variables categóricas
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:  # Para variables numéricas
            df[column].fillna(df[column].mean(), inplace=True)
            
    # Si hay errores, devolverlos antes de continuar
    if errors:
        raise ValueError(f"Data validation errors: {errors}")
    
    df_grouped = df.groupby('COD_ALUMNO').agg({
        'SEXO': 'first',  # Sexo no cambia, tomamos el primero
        'EDAD': 'last',  # Fecha de nacimiento tampoco cambia
        'ESTADO': 'last',  # Tomar el último estado
        'SEM_CURSADOS': 'max',  # Semestres cursados totales
        'CANT_RESERVAS': 'sum',  # Total de reservas de cursos
        'NOTA': 'mean',  # Promedio de nota
        'APROBO': lambda x: (x == 'S').mean(),  # Proporción de cursos aprobados
        'CREDITOS': 'sum',  # Créditos totales cursados
        'HRS_INASISTENCIA': 'sum',  # Horas totales de inasistencia
        'HRS_CURSO': 'sum',  # Horas totales de curso
        'PRCTJE_INASISTENCIA': 'mean',  # Porcentaje promedio de inasistencia
        'PONDERADO': 'mean',  # Promedio ponderado final
        'BECA_VIGENTE': 'max',  # Si tuvo alguna beca
        'IDH': 'first',  # IDH del lugar de procedencia
        'POR_POBREZA': 'first',  # Porcentaje de pobreza del lugar de procedencia
        'POR_POBREZA_EXTREMA': 'first',  # Porcentaje de pobreza extrema del lugar de procedencia
        'TIPO_CURSO': lambda x: (x == 2).mean() # Proporción de cursos obligatorios
    }).reset_index()

    df_grouped = df_grouped.rename(columns={'TIPO_CURSO': 'PROP_OBLIGATORIOS'})
    df_grouped['CAMBIO_RENDIMIENTO'] = df.groupby('COD_ALUMNO')['NOTA'].transform(lambda x: x.iloc[-1] - x.iloc[0])

    # Redondear a 2 decimales
    df_grouped['NOTA'] = df_grouped['NOTA'].round(2)
    df_grouped['APROBO'] = df_grouped['APROBO'].round(2)
    df_grouped['PRCTJE_INASISTENCIA'] = df_grouped['PRCTJE_INASISTENCIA'].round(2)
    df_grouped['PONDERADO'] = df_grouped['PONDERADO'].round(2)
    df_grouped['PROP_OBLIGATORIOS'] = df_grouped['PROP_OBLIGATORIOS'].round(2)

    # Convertir a enteros
    df_grouped['COD_ALUMNO'] = df_grouped['COD_ALUMNO'].astype(int)
    df_grouped['SEM_CURSADOS'] = df_grouped['SEM_CURSADOS'].astype(int)
    df_grouped['CANT_RESERVAS'] = df_grouped['CANT_RESERVAS'].astype(int)
    df_grouped['CREDITOS'] = df_grouped['CREDITOS'].astype(int)
    df_grouped['HRS_INASISTENCIA'] = df_grouped['HRS_INASISTENCIA'].astype(int)
    df_grouped['HRS_CURSO'] = df_grouped['HRS_CURSO'].astype(int)
    df_grouped['BECA_VIGENTE'] = df_grouped['BECA_VIGENTE'].astype(int)

    # Crear la nueva columna 'y'
    df_grouped['y'] = df_grouped['ESTADO'].apply(lambda x: 1 if x in ['Egresado', 'Regular'] else 0)

    return df_grouped

def preprocess_data(df):
    df = group_and_transform(df)
    label_encoder_sexo = LabelEncoder()
    label_encoder_sexo.fit(df['SEXO'])
    # Realizar las mismas transformaciones que aplicaste a los datos originales
    df['SEXO'] = label_encoder_sexo.transform(df['SEXO'])
    
    # Seleccionar las mismas columnas relevantes
    X = df.drop(columns=['COD_ALUMNO', 'y', 'ESTADO', 'CANT_RESERVAS', 'POR_POBREZA', 'POR_POBREZA_EXTREMA'])
    y = df['y']
    return X, y, df

# Función para el pipeline de contrafactuales (usando el código existente)
def pipeline_contrafactuales(alumno, dataframe, model):
    alumno = pd.DataFrame(alumno)
    X_nuevos, y_nuevos, X_transformado = preprocess_data(alumno)
    
    dataframeX, dataframeY, dataframeTransformado = preprocess_data(dataframe)
    dataframeX = pd.concat([dataframeX, X_nuevos]).reset_index(drop=True)
    dataframeY = pd.concat([dataframeY, y_nuevos]).reset_index(drop=True)
    dataframeTransformado = pd.concat([dataframeTransformado, X_transformado]).reset_index(drop=True)

    pred_nuevos = model.predict(X_nuevos).reshape(1, -1)

    if pred_nuevos[0] == 0:
        dice_data_nuevo = dice_ml.Data(
            dataframe=dataframeX.join(dataframeY), 
            continuous_features=['APROBO', 'NOTA', 'PRCTJE_INASISTENCIA', 'PONDERADO', 'IDH', 'PROP_OBLIGATORIOS', 'CAMBIO_RENDIMIENTO'], 
            outcome_name='y'
        )
        dice_model = dice_ml.Model(model=model, backend="sklearn")
        alumno_ = dataframeX.iloc[len(dataframeX) - 1 : len(dataframeX)]
        dice_nuevo = Dice(dice_data_nuevo, dice_model)
        
        counterfactuals_nuevo = dice_nuevo.generate_counterfactuals(
            alumno_, 
            total_CFs=10, 
            desired_class=1,
            features_to_vary=['APROBO', 'NOTA', 'PONDERADO', 'BECA_VIGENTE', 'CAMBIO_RENDIMIENTO'],
            permitted_range={
                'APROBO': [alumno_['APROBO'].values[0], int(max(dataframeX['APROBO'].values)) - 0.3], 
                'NOTA': [alumno_['NOTA'].values[0], int(max(dataframeX['NOTA'].values)) - 5],
                'PONDERADO': [alumno_['PONDERADO'].values[0], int(max(dataframeX['PONDERADO'].values)) - 6.0],
                'CAMBIO_RENDIMIENTO': [alumno_['CAMBIO_RENDIMIENTO'].values[0], alumno_['CAMBIO_RENDIMIENTO'].values[0] + 2.0],
            }
        )
        # Convertir el DataFrame de los contrafactuales a JSON
        contrafactuales_json = counterfactuals_nuevo.cf_examples_list[0].final_cfs_df.to_json(orient='records')
        
        # Convertir el DataframeTransformado a JSON
        alumno_json = alumno_.to_json(orient='records')
        
        return json.loads(contrafactuales_json), json.loads(alumno_json)
    else:
        return {"message": "El alumno no fue clasificado como desertor."}

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/generate', methods=['POST'])
def generate_contrafactuals():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No se subió ningún archivo JSON"}), 400

    # Leer el archivo JSON
    alumno = json.load(file)
    # Suponiendo que el archivo de datos principal está cargado
    dataframe = pd.read_csv(os.path.join(os.getcwd(), "PRODUCTO DE DATOS")+"/Desercion_clean_enriched.csv")  

    # Generar los contrafactuales
    contrafactuales, alumno_  = pipeline_contrafactuales(alumno, dataframe, loaded_model)
    
    print(alumno_)
    
    print(contrafactuales)

    # Retornar como JSON
    return jsonify({"alumno": alumno_, "contrafactuales": contrafactuales})

@app.route('/search', methods=['POST'])
def search_contrafactual():
    # Leer el código del alumno desde el formulario
    cod_alumno = request.form.get('cod_alumno')
    
    # pasar a float
    cod_alumno = float(cod_alumno)
    
    print(cod_alumno)

    if not cod_alumno:
        return jsonify({"error": "Debe ingresar un código de alumno"}), 400

    try:
        # Cargar el dataframe principal
        dataframe = pd.read_csv(os.path.join(os.getcwd(), "PRODUCTO DE DATOS", "Desercion_clean_enriched.csv"))
        
        # Filtrar los datos del alumno basado en su código
        alumno_data = dataframe[dataframe['COD_ALUMNO'] == float(cod_alumno)]
        
        #pasar a diccionario con el formato
        '''
        {"COD_PERSONA": [10073.0, 10073.0, 10073.0, 10073.0, 10073.0], "COD_ALUMNO": [969696.0, 969696.0, 969696.0, 969696.0, 969696.0], "SEXO": ["M", "M", "M", "M", "M"], "PER_INGRESO": ["2005-01", "2005-01", "2005-01", "2005-01", "2005-01"], "ESTADO_CIVIL": ["S", "S", "S", "S", "S"], "TIPO_COLEGIO": ["P\u00fablico en Convenio", "P\u00fablico en Convenio", "P\u00fablico en Convenio", "P\u00fablico en Convenio", "P\u00fablico en Convenio"], "PTJE_INGRESO": [114.8, 114.8, 114.8, 114.8, 114.8], "CARRERA": ["INGENIER\u00cdA INDUSTRIAL", "INGENIER\u00cdA INDUSTRIAL", "INGENIER\u00cdA INDUSTRIAL", "INGENIER\u00cdA INDUSTRIAL", "INGENIER\u00cdA INDUSTRIAL"], "ESTADO": ["Separado", "Separado", "Separado", "Separado", "Separado"], "SEM_ALUMNO": [2.0, 2.0, 2.0, 2.0, 2.0], "SEM_CURSADOS": [2.0, 2.0, 2.0, 2.0, 2.0], "CANT_RESERVAS": [1.0, 1.0, 1.0, 1.0, 1.0], "PER_MATRICULA": ["2004-02", "2004-02", "2004-02", "2004-02", "2004-02"], "COD_CURSO": [886.0, 887.0, 885.0, 888.0, 890.0], "CURSO": ["Metodolog\u00eda del Estudio", "Matem\u00e1tica B\u00e1sica", "Introducci\u00f3n a la Vida Universitaria", "\u00c1lgebra Lineal y Geometr\u00eda Anal\u00edtica", "Qu\u00edmica 1"], "CREDITOS": [3.0, 3.0, 3.0, 5.0, 3.0], "TIPO_CURSO": [2, 2, 2, 2, 2], "COD_PLAN": [49.0, 49.0, 49.0, 49.0, 49.0], "COD_GRUPO": [3189.0, 3187.0, 3193.0, 3190.0, 3191.0], "GRUPO": ["IND-1B", "IND-1B", "IND-1B", "IND-1B", "IND-1B"], "NOTA": [7.26, 7.35, 10.2, 8.5, 9.0], "APROBO": ["N", "N", "N", "S", "S"], "HRS_INASISTENCIA": [5.0, 0.0, 0.0, 2.0, 3.0], "HRS_CURSO": [48.0, 64.0, 48.0, 50.0, 52.0], "PRCTJE_INASISTENCIA": [13.0, 13.0, 13.0, 13.0, 13.0], "PONDERADO": [7.7933, 7.7933, 7.7933, 7.7933, 7.7933], "CRED_GRADUACION": [217.0, 217.0, 217.0, 217.0, 217.0], "BECA_VIGENTE": [0.0, 0.0, 0.0, 0.0, 0.0], "NOTA_ENCUESTA_DOC": [0.0, 0.0, 0.0, 0.0, 0.0], "Departamento_Procedencia": ["AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA"], "Provincia_Procedencia": ["AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA"], "Distrito_Procedencia": ["AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA"], "Departamento_Residencia": ["AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA"], "Provincia_Residencia": ["AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA"], "Distrito_Residencia": ["AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA", "AREQUIPA"], "POBLACION": [10119, 10119, 10119, 10119, 10119], "IDH": [0.4906, 0.4906, 0.4906, 0.4906, 0.4906], "POR_POBREZA": [34.8, 34.8, 34.8, 34.8, 34.8], "POR_POBREZA_EXTREMA": [21.8, 21.8, 21.8, 21.8, 21.8], "EDAD": [17, 17, 17, 17, 17]}
        '''
        if alumno_data.empty:
            return jsonify({"error": f"No se encontraron datos para el alumno con código {cod_alumno}"}), 404
        
        alumno_data = alumno_data.to_dict(orient='list')
        
        print(alumno_data)
        
        # Hacer el preprocesamiento y obtener los contrafactuales
        contrafactuales, alumno_json = pipeline_contrafactuales(alumno_data, dataframe, loaded_model)

        return jsonify({"alumno": alumno_json, "contrafactuales": contrafactuales})

    except FileNotFoundError:
        return jsonify({"error": "No se encontró el archivo de datos principal"}), 500

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400


if __name__ == "__main__":
    app.run(debug=True)
