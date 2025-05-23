# Ejemplos de uso

Aquí se mostrarán ejemplos del uso de la app:

## Entrada

```json
{
  "title": "El presidente dijo que la tierra es plana",
  "body": "Una noticia publicada por portales anónimos asegura que..."
}
```
## Salida
json
```
{
  "alerts": [
    "🔴 ALTO RIESGO: Posible contenido falso o engañoso...",
    "Frase vaga detectada: 'portales anónimos'"
  ],
  "risk_level": "high"
}
```

![Captura](screenshots/ejemplo1.png)