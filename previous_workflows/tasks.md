**SET 01**
*Concepto: Modelo Predictivo de Zonas de NYC*
Hey Claude, a partir de este momento crearemos un workflow para hacer un data scraping y luego un ploteo para entrenar a un modelo de ML, el cual tendrá el objetivo de predecir el uso de una zona urbana de la ciudad de NY (y en el futuro de otra ciudad del mundo) mediante el analisis de las amenidades colindantes. Quiero primero discutir contigo los ejes a analizar para posteriormente seguir un pipeline similar al que hemos desarrollado en la carpeta de General-OSM-Scraper.

*Reglas a seguir antes de iniciar el proyecto:*
1. Trabajaremos en una carpeta distinta a General-OSM-Scraper, la cual llamaremos "Zone Finding" (o algo así). Y tendra el objetivo de establecer el pipeline de las columnas de nuestro dataset csv que usaremos parae ntrenar el modelo de ML.
2. Los steps, o notebooks que llevaremos a cabo serán archivos.ipynb que nos permitan conocer lo que estamos haciendo. Estos deberan tener una descripcion breve de lo que hace el tool y cuales clumnas incorpora.
3. Las columnas del csv se crearán en estos notebooks secuenciales mediante un orchestrator (el cual sera paso 00_orchestrator.ipynb). La secuencia debe seguir el orden de izquierda a derecha del dataset csv. 
4. Los notebooks se pueden agrupar por columnas similares, pero al final debo contar con un dataset con columnas secuenciales que correspondan al pipeline que hemos establecido.
5. Usaremos OSMnx principalmente para extraer estos datos. Puedes proponer y añadir librerías utiles a estos notebooks del pipeline, siempre corroborando y verificando que la información tenga sentido y sea fideligna.
6. Al final de cada tarea importante o pipeline de pasos (los cuales seran agregados a este archivo tasks.md) deberás hacer un update a CLAUDE.md y realizar un commit con los cambios realizados.
7. Todos los textos, anotaciones, deben ser en ingles. Lo unico en español es lo que yo te escriba en el chat o por aqui. Pero debe ser todo en ingles en lo que trabajemos.

*El dataset*
1. "C:\Users\gramo\Downloads\04a3b68d1695a8c12c14bc02dbd22a58.jpg" Es una fuente de inspiracion que tuve para este proyecto. Donde la ciudad de NYC se divide por "zonas turisticas", o identificables por su programa urbano. Esto quiere decir que hay zonas que se caracterizan por ser "industriles" o "escolares" debido a su programa predominante. 
2. La relevancia de este proyecto es que puede ser entrenado en NYC (una ciudad caótica, pero muy completa en datos) y que luego pueda predecir las zonas urbanas de otras ciudades del mundo.
3. Las columnas que tengo pensadas para predecir este tipo de zonas van desde:
    - ¿Cuál es el programa predominante?
    - ¿Que tipo de zona es? (puede servirnos para identificar el ground truth, sin embargo dudo que OSMnx diga la realidad, para ello puede que PLUTO sea una mejor fuente, idk tbh lol)
    - ¿Cuál es el Median income de la gente que habita, o visita estos sitios?
    - ¿Cuál es la intensidad del turismo en la zona?
    - *Agrega o descarta temas que sean relevantes (o no) para esta infestigación.
4. Mi compañero de equipo propone estas formulas, no estoy seguro de que puedan ser correctas, als dejo aqui de todas formas:
    - Tourist shops = high tourism_intensity area + high median_income + commercial zone
    - Local shops = low tourism_intensity + residential zone + lower income
5. Mi compañero propone tambien esto para hace rla investigacion, consideralo o aclarame si vale la pena dedicarle tiempo: The Y variable would come from PLUTO landuse column — already in your downloaded file. No new data needed.

* Identifia las pros y contras de mi proyecto y propón soluciones antes de iniciar el workflow exhaustivo. Hazme preguntas, aclaraciones o cualquier otro tema para tomar el mejor rumbo de acuerdo al tipo de datos que buscaremos. Enfocate en las mejores decisiones que puedan garantizar relaciones y predicciones entre los datasets de un modelo de classification.


**SET 02**
*aqui estrán debajo los tasks que hay que trabajar numeradas y separadas por sets mediante una line, las instrucciones las tendrás en el chat*
1. Haz un notebook donde pueda visualizar en un mapa 2D los census tracts de mi csv. file combined generado. Descarga las librerías generadas ahi para lograr su visualización.
2. Crea una copia del notebook NYC_classification.ipynbm donde tenga el nombre de mi proceso Zone-Finding en el nombre del archivo. Ajusta los parametros para generar los plots necesarios mediante mi archivo combinado del dataset de mi Zone-Finding pipeline.
3. CAMBIA EL NOTEBOOK DEL STEP 10_ML_CLASSIFICATION POR UNA COPIA DEL NYC_classification.ipynbm (Ese es mi proceso de ML QUE SE QUE FUNCIONA). Ajusta los parametros para generar los plots necesarios mediante mi archivo combinado del dataset de mi Zone-Finding pipeline          

3. Optimiza 05_Accessibility mediante tus sugerencias:
    - Hacer 1 query batch para todo Manhattan en vez de 310 queries individuales — buscar todos los subways y bus stops de Manhattan en un solo query, luego calcular distancias localmente con BallTree
    - Reducir el sleep a 0.3s para queries cacheados (ya no se necesita rate limit)