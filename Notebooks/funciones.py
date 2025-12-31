import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

def test_mann_whitney(df, col_metrica, col_grupo, val_grupo1=1, val_grupo0=0, alpha=0.05, alternative='two-sided'):
    """
    Ejecuta el Test U de Mann-Whitney a un nivel de confianza personalizado con un tipo de test a elección. 
    Toma una columna numérica (col_métrica) y las agrupa por una columna binaria (col_grupo). Los valores de agrupación
    se indican en val_grupo0 y val_grupo1.
    
    Parámetros:
    - df: DataFrame con los datos.
    - col_metrica: Columna numérica a analizar (ej: 'wards_per_min', 'salario').
    - col_grupo: Columna binaria que separa los grupos (ej: 'win', 'genero').
    - val_grupo1: Valor que define al 'Grupo de Interés' o Grupo 1 (ej: 1, 'Mujer').
    - val_grupo0: Valor que define al 'Grupo de Referencia' o Grupo 0 (ej: 0, 'Hombre').
    - alternative: 'two-sided', 'greater', 'less'.
      * greater = Grupo A > Grupo B.
      * less = Grupo A < Grupo B.
      * two-sided = Grupo A =! Grupo B.
      
    """
    
    # 1. Validación de columnas
    if col_metrica not in df.columns or col_grupo not in df.columns:
        print(f"Error: Columna '{col_metrica}' o '{col_grupo}' no encontrada.")
        return

    # 2. Separación de Grupos Dinámica
    # Usamos los valores pasados como argumentos (val_grupo1, val_grupo0)
    data_g1 = df[df[col_grupo] == val_grupo1][col_metrica].dropna()
    data_g0 = df[df[col_grupo] == val_grupo0][col_metrica].dropna()

    if len(data_g1) == 0 or len(data_g0) == 0:
        print(f"Error: No hay datos para el grupo '{val_grupo1}' o '{val_grupo0}'.")
        return

    # 3. Cálculo de Estadísticos
    media_g1 = data_g1.mean()
    media_g0 = data_g0.mean()
    
    # 4. Ejecución del Test
    stat, p_value = mannwhitneyu(data_g1, data_g0, alternative=alternative)

    # 5. Reporte Agnóstico (Sin términos de negocio)
    print(f"📊 TEST MANN-WHITNEY: '{col_metrica}' por '{col_grupo}'")
    
    if alternative == 'greater':
        type_test = 'mayor'
        print(f"   H1 (Hipótesis Alternativa): Grupo '{val_grupo1}' es {type_test.upper()} que Grupo '{val_grupo0}'")

    elif alternative == 'less':
        type_test = 'menor'
        print(f"   H1 (Hipótesis Alternativa): Grupo '{val_grupo1}' es {type_test.upper()} que Grupo '{val_grupo0}'")

    else:
        type_test = 'diferente'
        print(f"   H1 (Hipótesis Alternativa): Grupo '{val_grupo1}' es {type_test.upper()} del Grupo '{val_grupo0}'")
    
    
    # 6. Interpretación Estadística Pura
    if p_value < alpha:
        print(f"   ✅ RESULTADO SIGNIFICATIVO (p < {alpha})")
        print(f"   -> Se rechaza la Hipótesis Nula (H0).")
        
        if alternative == 'greater':
            print(f"   -> Hay evidencia de que '{col_metrica}' es MAYOR en el grupo '{val_grupo1}'.")
        elif alternative == 'less':
            print(f"   -> Hay evidencia de que '{col_metrica}' es MENOR en el grupo '{val_grupo1}'.")
        else:
            print(f"   -> Hay evidencia de una diferencia en la distribución entre grupos.")
    else:
        print(f"   ❌ NO SIGNIFICATIVO (p >= {alpha})")
        print(f"   -> No se puede rechazar la Hipótesis Nula (H0).")
        print(f"   -> No hay evidencia suficiente para afirmar la relación planteada.")
    
    print("-" * 60 + "\n")
    return stat, p_value
    
# -------------------------------------------------------------------------------------------

def scaler(X_train, X_test, col_excluidas: list = []):
    
    '''
    X_train: DataFrame de X para el entrenamiento.
    X_test: DataFrame de X para el test.
    col_excluidas: Todas las columnas que no se deben scalar de X train y X test.
    
    Toma X_test y X_train para scalar las columnas que nosotros indiquemos previamente. 
    Se debe excluir las columnas que sean categóricas.
    '''
    # Guardamos los nombres de columnas a incluir en el scaler
    columnas_incluir = X_train.columns.difference(col_excluidas)
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    # Escalamos X_test y X_train
    sc = StandardScaler()
    X_train_scaled[columnas_incluir] = sc.fit_transform(X_train_scaled[columnas_incluir])
    X_test_scaled[columnas_incluir] = sc.transform(X_test_scaled[columnas_incluir]) # Solo aplicamos transform para evitar leak de datos.

    print(f'Se escalaron las siguientes columnas {list(columnas_incluir)}')

    return X_train_scaled, X_test_scaled

#---------------------------------------------------------------
def obtener_pares_correlacionados(df, umbral=0.7):
    """
    Encuentra pares de variables con correlación >= |umbral| (por defecto 0.7).
    
    Parámetros:
    - df: DataFrame de pandas (solo columnas numéricas serán consideradas).
    - umbral: Valor absoluto mínimo de correlación (entre 0 y 1).
    
    Retorna:
    - DataFrame ordenado con columnas: 'Variable 1', 'Variable 2', 'Correlación'
    """
    # Seleccionar solo columnas numéricas
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Calcular matriz de correlación
    corr_matrix = df_numeric.corr()
    
    # Filtrar correlaciones >= |umbral| y evitar duplicados (A-B vs B-A)
    pares = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= umbral:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                pares.append((col1, col2, corr))
    
    # Ordenar por valor absoluto de correlación (de mayor a menor)
    pares.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Convertir a DataFrame
    df_resultado = pd.DataFrame(pares, columns=['Variable 1', 'Variable 2', 'Correlación'])
    
    return df_resultado

#---------------------------------------------------------------
def replace_nan_bin(df,target,percent_clean):
    '''
    Dividimos el dataframe en target 1 y 0 y eliminamos los datos nulos o reemplazamos por la media según 
    los criterios impuestos en el desafío.
    '''
    
    df_0 = df.query(f"{target} == 0") # Filtramos por target 0
    
    # Vemos cuantos nulos existen
    for col in df_0.columns:
        nan_col = df_0[col].isna().sum()
        percent_na = nan_col / len(df) * 100
    
        
        if percent_na < percent_clean: # Eliminamos nulos si es menor a 1% y si es mayor, reemplazamos.
            if nan_col == 0:
                print(f"No existen valores nulos para la columna {col} cuando target vale 0")
                continue
                
            # Eliminar filas con NaN
            df_0 = df_0.dropna(subset=[col])
            final_nan = df_0[col].isna().sum()
            print(f"En la columna {col} se eliminaron {nan_col} registros. Quedan {final_nan} registros nulos")
        else:
            # Llenar NaN con la media de la columna
            mean_0 = df_0[col].mean()
            df_0[col] = df_0[col].fillna(mean_0)
            final_nan = df_0[col].isna().sum()
            print(f"En la columna {col} se reemplazaron {nan_col} registros. Quedan {final_nan} registros nulos")
    
    # Hacemos el mismo proceso pero para el filtro de target 1
    df_1 = df.query(f"{target} == 1")
    for col in df_1.columns:
        nan_col = df_1[col].isna().sum()
        percent_na = nan_col / len(df) * 100
    
        if percent_na < percent_clean:
            if nan_col == 0:
                print(f"No existen valores nulos para la columna {col} cuando target vale 1")
                continue
                
            # Eliminar filas con NaN en esta columna (usando DataFrame.dropna)
            df_1 = df_1.dropna(subset=[col])
            final_nan = df_1[col].isna().sum()
            print(f"En la columna {col} se eliminaron {nan_col} registros. Quedan {final_nan} registros nulos")
        else:
            # Llenar NaN con la media de la columna
            mean_1 = df_1[col].mean()
            df_1[col] = df_1[col].fillna(mean_1)
            final_nan = df_1[col].isna().sum()
            print(f"En la columna {col} se reemplazaron {nan_col} registros. Quedan {final_nan} registros nulos")
    
    # Combinar los DataFrames corregidos
    df_sin_nan = pd.concat([df_1, df_0], axis=0)
    return df_sin_nan
#------------------------------------------------------------------------------------------------------------------
def hora_to_datetime(df,columna_datetime):
    '''
    A partir del dataframe entregado, se formatean a h:m:s las columnas indicadas
    '''
    for col in columna_datetime:
        # Convertir la columna 'hora' a datetime, pero manteniendo solo la hora
        df_[col] = pd.to_datetime(df[col], format='%H:%M:%S').dt.time

    return df
#---------------------------------------------------------------           
def borrar_atipicos_IQR(data, col_select: 'lista'=None,ignore_umbral=False):
    
    '''
    Data: DataFrame.
    col_select: Columna del dataframe para quitar valores atípicos. (lista)
    ignore_umbral: ignorar o no umbral del 5% de los datos para decidir si eliminarlos o no.
    
    Calcula los límites inferiores y superiores de las columnas indicadas de un dataframe 
    usando IQR, muestra el número de datos atípicos y el porcentaje de datos respecto al total, 
    y elimina los registros con valores atípicos si el porcentaje es menor al 5% (si ingore_umbral es True, solo los borra).
    '''
    
    data = data.copy()

    # Aseguramos que sea una lista para poder iterar, incluso si es solo una columna
    if col_select is None:
        col_select = [] # O manejar error
    else:
        col_select = list(col_select)
    
    # Mostrar las columnas seleccionadas
    print("Columnas seleccionadas:")
    print(col_select)
    print("Revisando valores atípicos...")
        
    resultados = []  # Lista para almacenar los resultados por columna
    
    # Itera sobre todas las columnas que no están en 'drop_columns'
    for columna in col_select:
        Q1 = data[columna].quantile(0.25)
        Q3 = data[columna].quantile(0.75)
        IQR = Q3 - Q1

        li = Q1 - 1.5 * IQR  # Límite inferior
        ls = Q3 + 1.5 * IQR  # Límite superior

        # Filtra los datos no atípicos para esta columna
        registros_atipicos = data[(data[columna] < li) | (data[columna] > ls)]  # DF de los datos atípicos
        total_atipicos_col = len(registros_atipicos)                            # Cantidad registros atípicos
        percent_atipicos_col = (total_atipicos_col / len(data[columna])) * 100  # Porcentaje de valores atípicos respecto al total de datos
    
        print(f"\nEl total de datos atípicos en '{columna}' es: {total_atipicos_col}. Representan el {percent_atipicos_col:.2f}% de los datos.")
        
        # Se evalua si se ingora o no el umbral del 5% para saber si se borran o no los valores outliers
        if ignore_umbral == True:
            print("Se eliminaron los registros con valores atípicos.")
            data = data[(data[columna] >= li) & (data[columna] <= ls)]
        else:
            # Si el porcentaje de atípicos es menor al 5% y no se ignora el umbral, eliminamos los valores atípicos
            if percent_atipicos_col < 5:
                print("El porcentaje de atípicos es menor al 5%. Eliminando los registros con valores atípicos...")
                data = data[(data[columna] >= li) & (data[columna] <= ls)]
            else:
                print("El porcentaje de atípicos es mayor o igual al 5%. No se eliminaron registros.")

        # Almacena los resultados obtenidos
        resultados.append({"nombre_columna": columna, 
                           "datos_atipicos": total_atipicos_col, 
                           "porcentaje": percent_atipicos_col,
                           "limite_inf": li,
                           "limite_sup": ls})
    resultados_atipicos= pd.DataFrame(resultados)

    # Devuelve el data frame eliminando los atípicos si es que los hay
    return resultados_atipicos, data 
#--------------------------------------------------------------- 

def clasificar(cadena):
    '''
    clasifica la cadena en 0,1 y 2
    '''
    if cadena[0] == '1':
        return 1
    elif cadena[0] == '2':
        return 2
    else:
        return 0
#---------------------------------------------------------------

def limpieza(df,drop,col_horas,col_clas_dummies,columna_datetime,col_replace,col_extract_horas_func):
    '''
    Limpia el data frame entregado, cambiando los formatos de las columnas indicadas y quita outliyer en caso de requerirlo 
    Devuelve un dataframe limpio.
    '''

    def count_nulos(df):
        columnas = df.columns
        for col in columnas:
            num_null = df[col].isna().sum()
            print(f"{col} tiene:\n{num_null} valores perdidos")
            print("-"*20)
        return 
    
    count_nulos(df)
    print("."*20)
    
    df = df.drop(columns = drop)

    def extract_horas(df,col_extract_horas_func):
        
        for col in col_extract_horas_func: 
            print(f"En base a {col}, cómo llamará la nueva columna :")
            new_name = input("")
            # Convertir la columna de tiempo en formato HH:MM:SS a datetime
            df[col] = pd.to_datetime(df[col], format='%H:%M:%S')
            
            # Calcular las horas totales en formato decimal y redondear a dos decimales
            df[new_name] = (df[col].dt.hour + df[col].dt.minute / 60.0).round(2)
            df[new_name] = df[new_name].astype(float)
            df = df.drop(columns=[col])
        print("Creando nuevas columnas...")
        return df
        
    df = extract_horas(df,col_extract_horas_func)
    print("."*20)
           
    def convertir_a_horas(df, col_horas):
        
        # Normalizamos las horas para que siempre tengan dos dígitos
        df['horas'] = df[col_horas].str.extract(r'(\d{1,2})h')[0].str.zfill(2)  # Extraemos las horas y aseguramos 2 dígitos
        df['minutes'] = df[col_horas].str.extract(r'(\d{1,2})m')[0]  # Extraemos los minutos, rellenamos NaN con 0
        
        # Convertimos las horas a float
        df['horas'] = df['horas'].astype(float)
        df['minutes'] = df['minutes'].astype(float)
        
        # Juntamos ambas columnas para dejar el tiempo en horas
        df['Horas_de_vuelo'] = df['horas'] + df['minutes'] / 60
    
        df_horas = df.drop(columns = [col_horas,'horas','minutes'])
        df_horas['Horas_de_vuelo'] = df_horas['Horas_de_vuelo'].round(2)
        return df_horas

    df_horas = convertir_a_horas(df, col_horas)

    df_horas[col_clas_dummies[0]] = df_horas[col_clas_dummies[0]].apply(clasificar)
    
    
    def dummies(df, col_clas_dummies):
        num_col = len(col_clas_dummies)
        print(f"Existen {num_col} columnas a dummizar")
        
        # Inicializamos df_final como una copia del df original
        df_final = df.copy()
        
        for col in col_clas_dummies:
            print(f"Dummizando la columna {col}")
            
            # Generamos los dummies
            df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_dummies = df_dummies.astype(int)
            
            # Concatenamos los dummies al dataframe final
            df_final = pd.concat([df_final, df_dummies], axis=1)

        # Eliminamos las columnas originales que se dummizaron
        df_final = df_final.drop(columns=col_clas_dummies)
        
        print(f"Columnas finales: {df_final.columns}")
        
        return df_final

    df_final = dummies(df_horas,col_clas_dummies)
    print("."*20)

    def reemplazar(df,col_replace):
        df[col_replace] = df[col_replace].astype(str)  # Asegurar que es string
        df[col_replace] = df[col_replace].str.replace(',', '', regex=True)  # Eliminar comas
        df[col_replace] = pd.to_numeric(df[col_replace], errors='coerce')  # Convertir a número
        
        return df

    df_final = reemplazar(df_final,'price')

    df_limpio = borrar_atipicos_IQR(df_final)

    print("\nLimpiando tu data, espera...")
    print("-"*20)
    print("\nEste es tu data frame limpio:\n")
    print("-"*20)
    
    return df_limpio