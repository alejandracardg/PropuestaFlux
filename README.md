# Freixenet FLUX · simulador educativo

Simulador de la propuesta **Freixenet FLUX** (Diseño Organizativo, UC3M). El presentador controla el escenario desde el panel lateral o cede el mando a una persona que escanea un QR; la pantalla reorganiza la estructura en directo.

> Simulación educativa con **datos sintéticos**. No usa datos de Freixenet.

## Qué hace
- **Modelo de IA real**: gradient boosting (scikit-learn) entrenado con 6.000 campañas sintéticas. Estima el riesgo de que la demanda supere al producto disponible (AUC ≈ 0,97 en validación). Recomienda; no decide.
- **Motor de reglas transparente** (`engine.py`): traduce el riesgo y el escenario en una de cinco configuraciones, agrupadas en los tres modos del trabajo:
  - Modo 1 · Situación estable: solo la jerarquía funcional.
  - Modo 1 · Vigilancia: hay señales de cambio; se refuerza la comunicación lateral entre Comercial, Operaciones y Logística sin crear célula.
  - Modo 2 · Célula temporal: las áreas que entran dependen de la causa (Comercial si hay pedidos que priorizar, Logística si hay que mover producto o hay incidencias, Finanzas si las alternativas tienen coste).
  - Modo 2 · Célula con enlace al grupo: si afecta a varios mercados sin ser grave, entra Marketing internacional y una posición de enlace con Henkell sin autoridad formal.
  - Modo 3 · Equipo conjunto Freixenet–Henkell: problema grave en varios mercados; decide el grupo.
  Con incertidumbre alta se incorpora Análisis de datos (tecnoestructura) y el ritmo de coordinación pasa de semanal a diario.
- **Comparación tradicional vs. FLUX** con supuestos explícitos y editables desde el panel lateral (horas por nivel jerárquico, duración de la reunión de la célula, efecto de la incertidumbre). En el modo 1 FLUX no crea célula y da los mismos resultados que el modelo tradicional.

## Dos vistas
- Proyector: `https://<tu-app>.streamlit.app/`
- Mando externo (móvil): `https://<tu-app>.streamlit.app/?vista=mando`

En el panel lateral del proyector se elige **quién controla el escenario**:
- **Manual** (por defecto): el presentador mueve los controles del panel lateral. No necesita wifi en el móvil.
- **Mando externo (QR)**: aparece un QR; quien lo escanea controla el escenario desde su móvil y cada cambio se ve en la pantalla en unos 2 segundos. Si lo escanean varias personas, manda el último cambio.

## Publicarlo gratis (Streamlit Community Cloud, unos 10 minutos)
1. Crea un repositorio en GitHub y sube estos archivos: `app.py`, `engine.py`, `requirements.txt` y la carpeta `.streamlit/`.
2. Entra en https://share.streamlit.io con tu cuenta de GitHub → **Create app** → elige el repositorio, rama `main` y archivo `app.py`.
3. En **App URL / custom subdomain** escribe `freixenet-flux`. Así la dirección será `https://freixenet-flux.streamlit.app`, que es la del QR de la diapositiva 17 (`/?vista=mando`).
   - Si ese nombre está ocupado, elige otro, añade en *Settings → Secrets* la línea `APP_URL = "https://tu-nombre.streamlit.app"` y usa el QR que aparece en el panel lateral (o pídenos regenerar el de la diapositiva).
4. **Antes de clase**, abre la app unos minutos antes: las apps gratuitas se duermen si nadie las usa y tardan en despertar.

## Probarlo en local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Para votar desde móviles en local, deben estar en la misma red y usar la *Network URL* que muestra Streamlit.
