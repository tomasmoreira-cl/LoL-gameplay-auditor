# 🏆 Auditoría de Gameplay en League of Legends: Optimización de Rendimiento basada en datos

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Phase%202%20Completed-green)
![Type](https://img.shields.io/badge/Type-Data%20Scientist-orange)

> **🚧 Proyecto Finalizado** . Fase de Análisis Descriptivo e Inferencial completada; Modelado Predictivo completado.

## 🎯 Objetivo del Proyecto
Realizar una **auditoría basada en datos** del rendimiento personal en *League of Legends* de mis partidas y detectar las variables más predictorias en el resultado de las partidas. El proyecto utiliza un enfoque híbrido:

1.  **Análisis estadístico e inferencial:** Para identificar patrones tácticos de gameplay (gestión de oro, KDA, visión, champion pool, etc) y análisis de patrones del matchmaking mediante validación estadística.
2.  **Machine Learning (Predicción):** Para entrenar modelos supervisados capaces de **clasificar resultados y predecir victorias**, determinando matemáticamente qué variables (features) tienen mayor peso predictivo hasta el minuto 15 de las partidas.

## 📂 Contenido del Repositorio

El flujo de trabajo y las conclusiones detalladas se encuentran en la carpeta `notebooks`:

| Archivo | Descripción |
| :--- | :--- |
| `01_ETL.ipynb` | **Extracción ETL:** Script de conexión a la Riot API para descargar historial de partidas Flex y SoloQ. Crea un archivo CSV |
| `02_EDA.ipynb` | **Análisis Principal:** Limpieza, Tests de Hipótesis y generación del dataset maestro, análisis de correlaciones, selección de variables. **Contiene el diagnóstico de Visión, Rachas de victoria/derrotas, matchmaking y champion pool.** |
| `03_Modelamiento.ipynb` | **Modelado predictivo:** Script que ejecuta diferentes **modelos predictivos** (regresión, SVM, Red neuronal, Random forest, etc). Contiene un análisis para cada modelos revisando sus diferentes métricas (gap de aprendizaje, accuracy, f1-score, roc auc, etc) en pos de encontrar el mejor modelo y detectando las variables de mayor contribución a la predicción de los resultados.|
| `funciones.py` | **Librería Auxiliar:** Funciones reutilizables para tests estadísticos, limpieza de datos, etc. |

---

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.9+
* **Librerías:** Pandas, NumPy, SciPy (Estadística Inferencial), Matplotlib, Seaborn, Scikit-Learn, TensorFlow, etc.
* **Modelos:** Regresión logística, SVM, árboles de decisión, Random forest, XGB, red neuronal multicapa.
* **Datos:** Riot Games API (Match V5).

----

## 📊 Hallazgos Clave EDA y análisis inferencial (Fase 1)

### 1. Auditoría gestión de recursos de visión
Detecté una **asimetría crítica** entre mi inversión en visión y el win ratio:
* **En Flex Queue:** Identifiqué una visión ineficiente. Estoy invirtiendo en visión muy por encima del estándar de Diamante/Master. Esta inversión tiene un **Retorno Nulo ($p > 0.05$)** en mis victorias. **Mi Diagnóstico:** Estoy desperdiciando oro en visión que no estoy aprovechando, lo que retrasa mis picos de poder de combate.
* **En Solo Queue:** Mi estrategia de "Calidad sobre Cantidad" funciona en rol de Soporte, pero descubrí un **Superávit Estructural en el resto de roles**, donde mi volumen es muy superior al estándar competitivo, afectando negativamente mi control de mapa en un entorno donde la visión sí probó ser significativa para mí **($p > 0.01$)**.

Aun así, en general, la cantidad y la calidad son significativas. Por lo que mejorar ambas métricas en ambas colas debe ser prioritario.

### 2. Auditoría elección champion pool y roles main (SOLOQ y FLEX)
* **Roles main y champion pool:** Enfócate en los roles de Mid (Leblanc, Malzahar y Ahri) y Soporte (Thresh) en SoloQ; en caso de ir jungla solo usa Trundle. En Flex, Jungla o Top: Sejuani y Malphite (buena proyección). En cualquier caso, evita a toda costa el rol de soporte en Flex y si llegases a ir ADC, solo usa Ezreal. Estos picks me permitirán mejorar mis win ratios y romper la resistencia del matchmaking.

### 3. La Barrera del matchmaking (el "muro de habilidad")
Mi análisis de elasticidad de Matchmaking desmintió la existencia de una "Loss Queue" punitiva en Flex; luego de 2 derrotas el sistema intenta ayudar en recuperar tu MMR. Sin embargo, en SoloQ esto ocurre un poco diferente: se presenta un "salvavidas" ante 2 derrotas consecutivas pero en caso de perder, la probabilidad de ganar baja abruptamente en los siguientes 2 juegos. Por otro lado, el análisis confirma que me enfrento a un **Techo de Habilidad Rígido** luego de una racha de 3 victorias consecutivas, pero al pasar este techo, la probabilidad de ganar es mucho más alta.

* **El fenómeno de la racha de 3:** Tras una racha de 3 victorias consecutivas, mi rendimiento individual no decae abruptamente (KDA) pero sí colapsa ante variables como el oro y el XP. El aumento de MMR me pone a un mejor rival en frente pero no me gana a mí especialmente, sino por rotaciones, lecturas mejores de macrogame o aprovechando mejor los errores de mi equipo.
* **La causa:** El aumento de MMR me pone a un mejor rival en frente pero no me gana a mí especialmente, sino por rotaciones, lecturas mejores de macrogame o aprovechando mejor los errores de mi equipo.
* **Juego estratégico en racha +3:** Luego de la 3ra victoria consecutiva, lo más recomendable es jugar de manera inteligente, crea jugadas con mejor lectura de macrogame, no es necesario matar al rival en línea, debes generar un mayor impacto en el mapa más que en la línea.

### 4. Matriz de Correlación
El análisis de correlación reveló lo siguiente:

1.  **En SoloQ:** Mis juegos se deciden en los primeros 15 minutos. La estrategia óptima es **Agresiva/Snowball**. Debo priorizar campeones de *Early/Mid Game* que puedan ganar línea y convertir ese oro en KDA rápidamente, ya que el sistema no perdona las desventajas económicas y de XP.

2.  **En Flex:** El juego es permisivo. La estrategia óptima es **Coordinada/Scaling**. Puedo permitirme perder línea levemente si eso garantiza mejores peleas de equipo (*Teamfighting*) tarde, ya que el oro temprano no dicta la sentencia final.

3.  En ambas colas priorizar el heraldo y la primera torre. Luego del KDA, ambos objetivos son los que más inciden en la victoria.

4.  **Gestión de Expectativas:** Debo asumir que tras una racha positiva, la siguiente partida tendrá una probabilidad base de victoria menor debido al ajuste de MMR. En esos casos asume que tus probabilidades son más bajas, pero juega inteligente, aprovecha los errores del rival y, en caso de perder, no te preocupes, toma un descanso y juega nuevamente; tener una racha de derrotas luego de un MMR Check no es común.
     
---

## 📊 Hallazgos clave del modelado (Fase 2)

### Resumen comparativo de modelos

| Modelo | F1 Score 0 (Derrota) | F1 Score 1 (Victoria) | Accuracy General | ROC-AUC | FN (Falsos Neg.) | FP (Falsos Pos.) | Features Importance Principales |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Reg. Logística** | 0.72 | 0.72 | 0.72 | 0.75 | 18 | 19 | First Herald, First Tower, Void Grubs (+) / Role Top, Jungle (-) |
| **SVM** | 0.71 | 0.68 | 0.70 | 0.75 | 22 | 18 | First Herald, Turret Plates, First Tower |
| **DT (Decision Tree)**| 0.68 | 0.56 | 0.63 | 0.68 | 34 | 15 | First Herald, Void Grubs, Turret Plates |
| **RF (Random Forest)**| 0.73 | 0.69 | 0.71 | 0.71 | 22 | 16 | First Tower, First Herald, Void Grubs, XP Diff min 15 |
| **MLP (Red Neuronal)**| 0.63 | 0.62 | 0.62 | 0.71 | 25 | 25 | Racha Previa, Primera Torre, Void Grubs |
| **XGB (XGBoost)** | **0.74** | **0.72** | **0.73** | **0.76** | **19** | **17** | First Herald, First Tower, Turret Plates, Void Grubs |

Sin lugar a dudas, el mejor modelo es el XGB con ajuste de hiperparámetros. Es el modelo que logra detectar mejor las victorias y derrotas con un accuracy del 0.73 y un AUC del 0.76 (identifica entre derrotas y victorias con mayor precisión). XGB logró mantener estables los falsos negativos y falsos positivos (aún es un modelo un poco pesimista pero redujo su optimismo respecto a la regresión logística) y logró definir mejor las variables que más contribuyen en la predicción.

**Análisis de variables con mayor contribución**

- **First Herald:** Asesinar el Heraldo es definitivamente la variable más predictora. A pesar de haber perdido importancia en otros modelos, en general asegura un rendimiento tal durante los primeros 15 minutos de juego que permite decidir con mayor precisión las victorias y derrotas. Esto implica que a pesar de que el heraldo pueda fallar (usarlo mal, fallar al tirar torre, etc.) es un generador de brecha entre los equipos.

- **First Tower:** En el análisis de correlaciones el heraldo tiene una correlación moderada con tirar la primera torre (0.29) y por sí mismas tienen incidencia en la victoria. Esto lo confirma XGB, ambas variables son las más predictoras lo que nos indica claramente que obteniendo el heraldo y primera torre, la posibilidad de ganar aumenta. Ya sea que el heraldo se obtenga a partir de un mejor control de mapa por haber tirado primera torre o que el heraldo logre tirar una primera torre, en ambos casos logra generar un snowball importante en el equipo. 

- **Turret plates:** Las placas destruidas también aparecieron como correlación moderada en las correlaciones. A diferencia de las dos primeras variables que generan oro global, las placas generan snowball individual. Esto genera un poder de predicción importante pero menor, dado que el efecto es más individual que global.

- **Void grubs:** Como última variable en torno a los objetivos, la obtención de los void grubs sirve levemente para obtener las placas de torre y con un poco más de fuerza, obtener la primera torre. Sin embargo, la correlación más fuerte con tirar la primera torre sigue siendo hacer el heraldo, por lo que asesinar void grubs es de utilidad, pero el heraldo casi dobla a esta variable en importancia.

- **min15_control_wards, min15_wards_killed:** Por otro lado, gana relevancia el control de visión en el mapa. Es mucho menos predictor, pero no deja de ser moderadamente importante para el modelo. Tener visión constante con centinelas de control y destruir los centinelas enemigos son las métricas más relevantes para mis victorias en relación al control de visión en el mapa.

- **min15_kills, min15_xp_diff, history_wr_champ, min15_cs:** Por último, el modelo considera variables de micro game: las kills que obtuve, la XP, mi win rate con el campeón que uso y el farmeo. Todas estas variables son de juego individual, son levemente relevantes a comparación con las decisiones de macro game que las superan con más del doble de importancia.

En definitiva, las variables predictoras más relevantes están en torno a objetivos importantes del early game: Heraldo y primera torre. Las variables de voids y placas ayudan de alguna manera a lograr estos objetivos principales, ya sea teniendo oro individual o contribuyendo poco a poco a tirar la primera torre. Por otro lado, la visión del mapa y el rendimiento individual no dejan de ser importantes, con menor impacto, pero con incidencia en la predicción. En resumen, para ganar tus partidas y generar snowball antes del minuto 15, enfócate en el heraldo, la primera torre y asesinar los void grubs. Asimismo, controla el mapa con wards de control y cuida tu KDA pero no te preocupes si vas mal, no es lo más relevante tener 20 asesinatos, sino obtener oro global destruyendo objetivos importantes que contribuyen a todo el equipo.

---
## 🚀 Instrucciones de Uso
1.  Clonar el repositorio.
2.  Instalar dependencias: `pip install -r requirements.txt`
3.  Ejecutar los notebooks en orden numérico para reproducir el pipeline.

---
**Autor:** Tomás Moreira | [LinkedIn](https://www.linkedin.com/in/tomas-moreira/)
