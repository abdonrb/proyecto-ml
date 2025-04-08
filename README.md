# Proyecto de Machine Learning "Global Sales".
![MPG Analysis Banner](https://www.velfix.es/wp-content/uploads/2021/12/Velfix-programa-ropa.png)

---

## 📋 Descripcion del proyecto
Realización de serie temporal y modelos de regresión.

---

El **DataSet de Merch Sales** ofrece un conjunto de datos excelente para analizar patrones de ventas de una tienda online, como que ciudad/genero es la que mas ordenes realiza etc. A traves de estos datos vamos a crear varios modelos de prediccion, para optimizar la gestion de stock, aplicar campañas de marketing personalizadas y tener una prevision de ventas.

**Objetrivos Principales:**
- Testear modelos de serie temporal tales como: Arima, Sarimax, Phrophet. 
- Testear modelos de regresión.

---

 ## 📂 Estructura del Proyecto  

```plaintext
├── data/                # Archivos del dataset  
├── models /             # Modelos realizados. 
├── notebooks/           # Jupyter Notebooks para el análisis exploratorio  y visualizaciones.
├── reports/             # Reporte y memoria del proyecto.
├── src/                 # Clases utilizadas para optimizar el proyecto. 
├── README.md            # Descripción del proyecto (este archivo)  
```

---

## 🛠️ Herramientas y Librerías Utilizadas  

| Herramienta      | Uso                                                                 |
|------------------|---------------------------------------------------------------------|
| **Python**     | Lenguaje principal para manipulación y análisis de datos.         |
| **Pandas**     | Limpieza, transformación y análisis de datos tabulares.           |
| **Matplotlib** | Creación de gráficos estáticos para visualización.                |
| **Seaborn**    | Visualización avanzada de datos con gráficos estadísticos.         |
| **Scipy**      | Pruebas estadísticas e inferencia avanzada.                       |
| **Sklearn**    | Libreria utilizada para los modelos de regresión|
| **Arima**    | Modelo utilizado para la serie temporal|
| **Sarimax**    |Serie temporal con variables exógenas |

---

## 📈 **Resultados Clave**📊 
Inicialmente, abordé el problema como una serie temporal, pero los resultados obtenidos no fueron los esperados. Por ello, opté por replantear el enfoque y tratarlo como un problema de regresión. Esta decisión tuvo un impacto muy positivo en el desempeño del modelo, logrando reducir el error cuadrático medio (RMSE) a 176,8 .

## **Conclusiones**
 
Aplicar un modelo de serie temporal no siempre es la opción más adecuada. Es fundamental explorar distintos enfoques, comparar sus resultados y ajustar el modelo que ofrezca el mejor rendimiento para el problema específico.

```
