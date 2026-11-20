"""Motor de Freixenet FLUX.

Tres piezas, como en el trabajo:
1. Un modelo de aprendizaje automático (entrenado con datos SINTÉTICOS) que estima el
   riesgo de desajuste entre demanda y disponibilidad.
2. Unas reglas de organización, transparentes, que traducen ese riesgo y el escenario en
   una configuración: quién participa, quién decide y hasta dónde.
3. Una comparación, con supuestos explícitos, entre un modelo jerárquico tradicional y FLUX.

Nada de esto usa datos de Freixenet: es una simulación educativa del diseño propuesto.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

INCIDENCIAS = ["Ninguna", "Leve", "Grave"]
INCERTIDUMBRE = ["Baja", "Media", "Alta"]


@dataclass
class Escenario:
    demanda: float = 0.0          # variación de la demanda frente a lo previsto, en %
    disponibilidad: float = 100.0  # producto disponible, en % de lo previsto
    incidencia: int = 0            # 0 ninguna, 1 leve, 2 grave
    mercados: int = 1              # número de mercados afectados
    incertidumbre: int = 0         # 0 baja, 1 media, 2 alta

    def vector(self) -> np.ndarray:
        return np.array([[self.demanda, self.disponibilidad, self.incidencia, self.mercados, self.incertidumbre]], dtype=float)


# --------------------------------------------------------------------------------------
# 1. Modelo de riesgo entrenado con datos sintéticos
# --------------------------------------------------------------------------------------
def generar_datos(n: int = 8000, semilla: int = 7):
    """Genera campañas ficticias y marca si hubo desajuste (demanda > producto disponible)."""
    rng = np.random.default_rng(semilla)
    demanda = rng.uniform(-30, 80, n)
    disponibilidad = rng.uniform(40, 120, n)
    incidencia = rng.choice([0, 1, 2], n, p=[0.55, 0.30, 0.15])
    mercados = rng.integers(1, 7, n)
    incertidumbre = rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2])

    ruido = np.array([0.05, 0.12, 0.22])[incertidumbre]
    demanda_real = (1 + demanda / 100) * rng.normal(1, ruido)
    perdida = np.where(incidencia == 0, 0.0, np.where(incidencia == 1, rng.uniform(0, 0.10, n), rng.uniform(0.10, 0.30, n)))
    perdida = perdida * (1 + 0.08 * (mercados - 1))
    # El plan de campaña incluye un 15 % de stock de seguridad sobre la demanda prevista.
    oferta_real = (disponibilidad / 100) * 1.15 * (1 - np.clip(perdida, 0, 0.6))
    desajuste = (demanda_real > oferta_real * 1.03).astype(int)

    X = np.column_stack([demanda, disponibilidad, incidencia, mercados, incertidumbre])
    return X, desajuste


@dataclass
class ModeloRiesgo:
    modelo: GradientBoostingClassifier
    auc: float
    n_entrenamiento: int

    def riesgo(self, e: Escenario) -> float:
        return float(self.modelo.predict_proba(e.vector())[0, 1])


def entrenar_modelo() -> ModeloRiesgo:
    X, y = generar_datos()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    modelo = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=7)
    modelo.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1])
    return ModeloRiesgo(modelo=modelo, auc=float(auc), n_entrenamiento=len(X_tr))


# --------------------------------------------------------------------------------------
# 2. Reglas de organización (aprobadas por el comité ejecutivo español en el diseño)
# --------------------------------------------------------------------------------------
UMBRAL_CELULA = 0.35      # riesgo a partir del cual se recomienda formar la célula
UMBRAL_GRAVE = 0.90       # riesgo que, con tres o más mercados, se considera problema grave
CAMBIO_DEMANDA = 25       # variación de demanda (%) que por sí sola activa la célula


AREAS = {
    "com": "Comercial",
    "ope": "Operaciones",
    "log": "Logística",
    "fin": "Finanzas",
    "mkt": "Marketing internacional",
}
RITMOS = ["seguimiento semanal", "seguimiento cada dos días", "seguimiento diario"]


@dataclass
class Configuracion:
    modo: int
    variante: str
    nombre: str
    riesgo: float
    motivos: list[str] = field(default_factory=list)
    miembros: list[tuple[str, str]] = field(default_factory=list)   # (área, por qué entra)
    lateral: bool = False        # comunicación lateral reforzada (vigilancia)
    enlace: bool = False         # posición de enlace con el grupo, sin autoridad
    analista: bool = False       # análisis de datos (tecnoestructura) se incorpora
    ritmo: str = ""
    decisiones: list[tuple[str, str]] = field(default_factory=list)

    @property
    def finanzas_en_celula(self) -> bool:
        return any(k == "fin" for k, _ in self.miembros)


def _miembros(e: Escenario, riesgo: float, modo: int) -> list[tuple[str, str]]:
    m = [("ope", "coordina la célula: conecta producción, existencias y entregas")]
    if abs(e.demanda) >= 10 or riesgo >= 0.5:
        m.append(("com", "hay pedidos que priorizar" if e.demanda >= 0 else "hay que recolocar el exceso de producto"))
    if e.incidencia >= 1 or e.disponibilidad < 95 or e.mercados >= 2:
        m.append(("log", "hay que mover producto o resolver la incidencia de transporte"))
    if e.disponibilidad < 85 or e.incidencia >= 1 or abs(e.demanda) >= 40:
        m.append(("fin", "las alternativas tienen coste (transporte urgente, aplazamientos)"))
    if e.mercados >= 2 or modo == 3:
        m.append(("mkt", f"afecta a {e.mercados} mercados y a los compromisos fuera de España"))
    return m


def configurar(e: Escenario, riesgo: float) -> Configuracion:
    motivos: list[str] = []
    varios_mercados = e.mercados >= 2
    grave = e.incidencia == 2 or (riesgo >= UMBRAL_GRAVE and e.mercados >= 3)
    ritmo = RITMOS[e.incertidumbre]

    if varios_mercados and grave:
        modo, variante = 3, "equipo"
        motivos.append(f"Afecta a {e.mercados} mercados y la situación es grave "
                       f"({'incidencia grave' if e.incidencia == 2 else f'riesgo del {riesgo:.0%}'}): "
                       "los compromisos entre países son decisiones corporativas.")
        motivos.append("Ante una crisis, la teoría prevé una centralización temporal en el grupo.")
    elif (riesgo >= UMBRAL_CELULA or abs(e.demanda) >= CAMBIO_DEMANDA or e.incidencia == 2
          or (e.incidencia == 1 and riesgo >= 0.20)):
        modo = 2
        variante = "enlace" if varios_mercados else "celula"
        if riesgo >= UMBRAL_CELULA:
            motivos.append(f"El modelo estima un riesgo de desajuste del {riesgo:.0%} (umbral: {UMBRAL_CELULA:.0%}).")
        if abs(e.demanda) >= CAMBIO_DEMANDA:
            motivos.append(f"La demanda cambia un {e.demanda:+.0f} % frente a lo previsto.")
        if e.incidencia >= 1:
            motivos.append(f"Hay una incidencia logística {INCIDENCIAS[e.incidencia].lower()}.")
        motivos.append("Las áreas pasan a depender unas de otras (interdependencia recíproca): conviene coordinarse por ajuste mutuo.")
        if variante == "enlace":
            motivos.append(f"Afecta a {e.mercados} mercados, pero no es grave: basta una posición de enlace con el grupo, "
                           "sin autoridad formal, para mantenerlo informado.")
    else:
        modo = 1
        vigilancia = riesgo >= 0.15 or abs(e.demanda) >= 10 or e.incidencia == 1 or e.incertidumbre == 2
        variante = "vigilancia" if vigilancia else "estable"
        if vigilancia:
            motivos.append(f"Hay señales de cambio (riesgo del {riesgo:.0%}), pero no justifican formar una célula.")
            motivos.append("Se refuerza la comunicación lateral entre áreas (ajuste mutuo) sin cambiar la jerarquía.")
        else:
            motivos.append(f"Riesgo bajo ({riesgo:.0%}) y sin incidencias relevantes: la estructura funcional es la más eficiente.")

    miembros = _miembros(e, riesgo, modo) if modo >= 2 else []
    analista = e.incertidumbre == 2 and modo >= 1 and variante != "estable"
    if analista:
        motivos.append("La incertidumbre de las previsiones es alta: se incorpora Análisis de datos (tecnoestructura) "
                       "para revisar la previsión y las reglas.")
    if modo >= 2:
        motivos.append(f"Ritmo de coordinación: {ritmo}, según la incertidumbre de las previsiones.")

    nombre = {"estable": "Modo 1 · Situación estable",
              "vigilancia": "Modo 1 · Vigilancia",
              "celula": "Modo 2 · Célula temporal",
              "enlace": "Modo 2 · Célula con enlace al grupo",
              "equipo": "Modo 3 · Equipo conjunto Freixenet–Henkell"}[variante]

    ejecutor = {1: "El departamento responsable", 2: "La célula (coordina Operaciones)", 3: "El equipo conjunto ejecuta"}[modo]
    decisiones = [
        ("Ajustes que respetan el presupuesto y las entregas confirmadas (p. ej., traslado entre almacenes)", ejecutor),
        ("Elegir entre clientes españoles con entregas confirmadas o superar el presupuesto delegado",
         "Comité ejecutivo español" + (" (la célula le presenta alternativas)" if modo == 2 else "")),
        ("Usar producto reservado para otro país o cambiar compromisos internacionales", "Grupo Henkell"),
    ]
    return Configuracion(modo=modo, variante=variante, nombre=nombre, riesgo=riesgo, motivos=motivos,
                         miembros=miembros, lateral=(variante == "vigilancia"), enlace=(variante == "enlace"),
                         analista=analista, ritmo=ritmo, decisiones=decisiones)


# --------------------------------------------------------------------------------------
# 3. Comparación tradicional vs. FLUX (supuestos explícitos, no datos reales)
# --------------------------------------------------------------------------------------
@dataclass
class Supuestos:
    horas_por_nivel: float = 8.0      # espera media por cada nivel jerárquico que hay que cruzar
    horas_celula: float = 3.0         # tiempo de una reunión de la célula
    penalizacion_incertidumbre: float = 0.5  # cada nivel de incertidumbre alarga la espera un 50 %


def comparar(e: Escenario, riesgo: float, s: Supuestos = Supuestos(), modo: int = 2) -> dict:
    """Estima, con supuestos explícitos, cómo se resuelven las incidencias de la campaña.

    En el modo 1 FLUX no activa ninguna célula: funciona igual que la estructura tradicional.
    """
    n = int(round(3 + 12 * riesgo + 3 * e.incidencia + (e.mercados - 1)))
    internacional = min(n, (e.mercados - 1) // 2 + (1 if e.incidencia == 2 and e.mercados >= 2 else 0))
    excepciones = min(n - internacional, int(round(n * (0.15 + 0.25 * riesgo))))
    locales = n - internacional - excepciones
    factor = 1 + s.penalizacion_incertidumbre * e.incertidumbre

    # Tradicional: todo lo que toca entregas sube por la cadena (área -> dirección -> comité),
    # y lo internacional, además, al grupo.
    niveles_trad = locales * 3 + excepciones * 3 + internacional * 4
    horas_trad = (locales * 3 + excepciones * 3 + internacional * 4) * s.horas_por_nivel * factor

    if modo == 1:
        trad_inc = [3 * s.horas_por_nivel * factor] * (locales + excepciones) + [4 * s.horas_por_nivel * factor] * internacional
        r = {"horas_media": horas_trad / n, "niveles_media": niveles_trad / n,
             "en_24h": 100 * sum(h <= 24 for h in trad_inc) / n}
        return {"incidencias": n, "locales": locales, "excepciones": excepciones, "internacionales": internacional,
                "tradicional": r, "flux": dict(r), "sin_celula": True}

    # FLUX: la célula resuelve lo local; solo suben las excepciones y lo internacional.
    niveles_flux = locales * 1 + excepciones * 2 + internacional * 2
    horas_flux = (locales * s.horas_celula + excepciones * (s.horas_celula + s.horas_por_nivel) +
                  internacional * (s.horas_celula + 2 * s.horas_por_nivel)) * factor

    def en_24h(horas_por_inc: list[float]) -> float:
        return 100 * sum(h <= 24 for h in horas_por_inc) / max(len(horas_por_inc), 1)

    trad_inc = [3 * s.horas_por_nivel * factor] * (locales + excepciones) + [4 * s.horas_por_nivel * factor] * internacional
    flux_inc = ([s.horas_celula * factor] * locales + [(s.horas_celula + s.horas_por_nivel) * factor] * excepciones +
                [(s.horas_celula + 2 * s.horas_por_nivel) * factor] * internacional)
    return {
        "incidencias": n, "locales": locales, "excepciones": excepciones, "internacionales": internacional,
        "tradicional": {"horas_media": horas_trad / n, "niveles_media": niveles_trad / n, "en_24h": en_24h(trad_inc)},
        "flux": {"horas_media": horas_flux / n, "niveles_media": niveles_flux / n, "en_24h": en_24h(flux_inc)},
        "sin_celula": False,
    }


# --------------------------------------------------------------------------------------
# Organigrama (DOT, renderizado por Streamlit)
# --------------------------------------------------------------------------------------
GOLD, GOLD_L, PANEL, WHITE, GREY = "#BC9A4C", "#CAB796", "#141414", "#F4F1EA", "#7A7A7A"


def organigrama(cfg: Configuracion) -> str:
    dim = "#3A3A3A"
    L = ["digraph G {", 'bgcolor="transparent"; rankdir=TB; splines=ortho; nodesep=0.45; ranksep=0.55; ordering=out;',
         f'node [shape=box style="rounded,filled" fillcolor="{PANEL}" color="{GREY}" fontcolor="{WHITE}" fontname="Lato" fontsize=15 penwidth=1.2 margin="0.2,0.09"];',
         f'edge [color="{GREY}" arrowsize=0.6 penwidth=1.1];']
    g3 = cfg.modo == 3
    L.append(f'henkell [label="Grupo Henkell" fillcolor="{GOLD if g3 else PANEL}" fontcolor="{"#050505" if g3 else WHITE}" color="{GOLD}"];')
    L.append(f'comite [label="Comité ejecutivo España" color="{GOLD}"];')
    L.append("henkell -> comite;")
    an_on = cfg.analista
    L.append(f'datos [label="Análisis de datos\n(tecnoestructura)" fontsize=13 '
             f'color="{GOLD if an_on else dim}" fontcolor="{WHITE if an_on else GREY}" style="rounded,filled{"" if an_on else ",dashed"}"];')
    L.append(f'comite -> datos [arrowhead=none color="{GOLD if an_on else dim}"];')
    L.append("{rank=same; comite; datos;}")
    miembros = dict(cfg.miembros)
    for k, t in AREAS.items():
        activo = k in miembros
        L.append(f'{k} [label="{t}" color="{GOLD if activo else GREY}"];')
        L.append(f"comite -> {k};")
    L.append("{rank=same; " + " ".join(AREAS) + ";}")

    if cfg.lateral:
        for a, b in [("com", "ope"), ("ope", "log")]:
            L.append(f'{a} -> {b} [dir=both style=dashed color="{GOLD}" constraint=false penwidth=1.6];')
        L.append(f'lat [shape=plaintext style="" label="comunicación lateral (ajuste mutuo)" fontcolor="{GOLD}" fontsize=13];')
        L.append('ope -> lat [style=invis];')

    if cfg.modo >= 2:
        etiqueta = ("Célula temporal · coordina Operaciones" if cfg.modo == 2 else "Equipo conjunto Freixenet–Henkell") + f" · {cfg.ritmo}"
        L.append(f'subgraph cluster_celula {{ label="{etiqueta}"; labelloc="b"; fontname="Lato"; fontcolor="{GOLD}"; fontsize=15; '
                 f'style="rounded,filled"; color="{GOLD}"; fillcolor="#1d1708"; penwidth=2;')
        for k in [a for a in AREAS if a in miembros]:
            nombre = {"mkt": "Marketing int."}.get(k, AREAS[k])
            L.append(f'r_{k} [label="{nombre}" fillcolor="{GOLD}" fontcolor="#050505" color="{GOLD}"];')
        if an_on:
            L.append(f'r_datos [label="Datos" fillcolor="{GOLD_L}" fontcolor="#050505" color="{GOLD}"];')
        if g3:
            L.append(f'r_grupo [label="Grupo Henkell" fillcolor="{GOLD_L}" fontcolor="#050505" color="{GOLD}"];')
        L.append("}")
        for k in miembros:
            L.append(f'{k} -> r_{k} [style=dashed color="{GOLD}"];')
        if an_on:
            L.append(f'datos -> r_datos [style=dashed color="{GOLD}"];')
        if g3:
            L.append(f'henkell -> r_grupo [color="{GOLD}" penwidth=1.8];')
        else:
            L.append(f'comite -> r_ope [style=dotted color="{GOLD}"];')
        if cfg.enlace:
            L.append(f'enlace [label="Enlace con el grupo\n(sin autoridad formal)" fontsize=13 color="{GOLD}" style="rounded,filled,dashed"];')
            L.append(f'henkell -> enlace [dir=both style=dashed color="{GOLD}"];')
            L.append(f'enlace -> r_ope [dir=both style=dashed color="{GOLD}"];')
    elif cfg.analista:
        pass
    L.append("}")
    return "\n".join(L)
