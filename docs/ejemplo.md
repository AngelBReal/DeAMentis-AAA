# Ejemplos y Demostraciones

Esta sección muestra el flujo completo de la aplicación **De A Mentis**, desde la interacción del usuario en la interfaz hasta el registro del resultado en el sistema de trazabilidad de MLflow.

---

## 1. Ingreso de Texto por el Usuario

El usuario accede a la interfaz web desplegada y encuentra un formulario simple para insertar el **título** y el **texto de una noticia**.

📸 **GIF 1** – Ingreso de texto y envío

![Ingreso de texto](docs\gif1.gif)

---

## 2. Proceso de Inferencia

Al hacer clic en “ Analizar ”, el backend realiza dos tareas:

### a) **Predicción con el modelo VotingClassifier**

El modelo retorna una probabilidad `prob_fake` (entre 0 y 1) de que la noticia sea falsa.

### b) **Análisis lingüístico educativo**

Un módulo aparte (`analyzer.py`) identifica patrones problemáticos como:

- Cifras sin fuente
- Citas ambiguas
- Sensacionalismo

Ambas salidas se fusionan para dar un resultado interpretativo.

📸 **GIF 2** – Visualización de resultados

![Resultados](docs\gif2.gif)

---

## 3. Interpretación de Resultados

Los resultados se presentan de forma educativa:

- Una **barra de riesgo** con codificación de colores:
  - Verde = bajo
  - Amarillo = moderado
  - Rojo = alto
- Una **lista de alertas** generadas por el analizador lingüístico
- Un mensaje claro que fomenta el pensamiento crítico (no un veredicto absoluto)

---

## 4. Registro en MLflow

Cada predicción es trazada en MLflow con:

- Probabilidad de falsedad (`prob_fake`)
- Alertas generadas
- Texto analizado
- Tiempo de respuesta

Esto permite:

- Validación post-despliegue
- Análisis estadístico de uso
- Auditoría pública si fuera necesario

 **GIF 3** – Registro en MLflow (interfaz web)

![Registro MLflow](docs\gif3.gif)

---

## 📁 Datos registrados por predicción

| Campo            | Descripción                                           |
|------------------|-------------------------------------------------------|
| `title`          | Título de la noticia enviada                          |
| `text`           | Texto completo de la noticia                         |
| `prob_fake`      | Probabilidad estimada por el modelo de ser falsa     |
| `riesgo`         | Nivel de riesgo interpretado según umbral            |
| `alertas`        | Lista de advertencias lingüísticas generadas         |
| `timestamp`      | Fecha y hora de la inferencia                        |

**Ejemplo Falso** 

![Falso](docs\png1.png)

**Ejemplo Verdadero** 

![Verdadero](docs\png1.png)
---

## ✅ Buenas prácticas

- Las alertas **no provienen del modelo**, sino de reglas lingüísticas predefinidas.
- El objetivo es **educar, no censurar**.
- La interfaz está diseñada para ser clara, respetuosa y accesible.

---

## ¿Cómo probarlo tú mismo?

Puedes acceder a la aplicación desplegada aquí:

🔗 [https://deamentis-frontend.onrender.com](https://deamentis-frontend.onrender.com)

