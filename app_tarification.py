# ============================================================
# APPLICATION DE TARIFICATION AUTOMOBILE RC
# Calculateur interactif de prime d'assurance
# Auteur : Thierry NIYOKWIZIGIRWA
# Master 2 Actuariat et Finance — Université du Burundi
# Technologie : Python + Streamlit + Plotly + Pandas
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================
# CONFIGURATION GENERALE DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="TariAuto — Calculateur de Prime RC",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONNALISE POUR UN DESIGN PROFESSIONNEL
# ============================================================
st.markdown("""
<style>
    /* Couleur de fond principale */
    .main {
        /*background-color: #f8f9fa;*/
        background-color: black;
    }
    
    /* Bandeau titre */
    .titre-principal {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        padding: 25px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Cartes de résultats */
    .carte-resultat {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 5px solid #2980b9;
        margin-bottom: 15px;
    }
    
    /* Carte verte pour bon profil */
    .carte-bon {
        border-left: 5px solid #27ae60;
    }
    
    /* Carte rouge pour profil risqué */
    .carte-mauvais {
        border-left: 5px solid #e74c3c;
    }
    
    /* Carte orange pour profil moyen */
    .carte-moyen {
        border-left: 5px solid #f39c12;
    }
    
    /* Métriques personnalisées */
    .metrique-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e3a5f;
    }
    
    /* Bouton principal */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(41, 128, 185, 0.4);
    }
    
    /* Footer */
    .footer {
        background: #1e3a5f;
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-top: 30px;
    }
    
    /* Badge niveau risque */
    .badge-faible {
        background: #27ae60;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .badge-moyen {
        background: #f39c12;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .badge-eleve {
        background: #e67e22;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .badge-tres-eleve {
        background: #e74c3c;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PARAMETRES DU MODELE GLM POISSON x LOG-NORMAL
# (Coefficients issus de l'analyse sur freMTPL2 — 677 991 polices)
# ============================================================

PRIME_PURE_REF = 80.59  # Prime pure du profil de référence (€)

CHARGEMENTS = {
    "Frais de gestion"    : 0.25,
    "Marge de sécurité"   : 0.10,
    "Coût de réassurance" : 0.05
}
CHARGEMENT_TOTAL = sum(CHARGEMENTS.values())  # 40%

# Relativités tarifaires (exp(coefficients GLM))
RELATIVITES = {
    "Area": {
        "A — Zone rurale"         : 1.000,
        "B — Zone semi-rurale"    : 1.122,
        "C — Zone semi-urbaine"   : 1.190,
        "D — Zone urbaine"        : 1.410,
        "E — Zone très urbaine"   : 1.503,
        "F — Métropole"           : 1.337
    },
    "DrivAge": {
        "18 — 25 ans" : 1.000,
        "26 — 35 ans" : 0.884,
        "36 — 45 ans" : 1.304,
        "46 — 55 ans" : 1.525,
        "56 — 65 ans" : 1.327,
        "65 ans et +"  : 1.551
    },
    "VehAge": {
        "0 — 1 an (neuf)"    : 1.000,
        "2 — 5 ans"          : 0.986,
        "6 — 10 ans"         : 0.968,
        "11 — 15 ans"        : 0.815,
        "16 — 25 ans"        : 0.697,
        "Plus de 25 ans"     : 0.363
    },
    "VehGas": {
        "Diesel"  : 1.000,
        "Essence" : 0.835
    },
    "VehBrand": {
        "Marque Autres" : 1.000,
        "Marque B1"     : 0.970,
        "Marque B2"     : 0.974,
        "Marque B3"     : 1.021,
        "Marque B4"     : 0.930,
        "Marque B5"     : 0.938,
        "Marque B6"     : 0.963,
        "Marque B10"    : 0.984,
        "Marque B12"    : 0.880
    }
}

# Coefficient BonusMalus
COEF_BM = 0.0265

# ============================================================
# FONCTIONS DE CALCUL
# ============================================================

def calculer_relativite_bm(bm):
    """Relativité BonusMalus centrée sur BM=50 (bonus max)"""
    return np.exp(COEF_BM * (bm - 50))

def calculer_prime(area, driv_age, veh_age, veh_gas, veh_brand, bm):
    """
    Calcule la prime pure et commerciale.
    Modèle : GLM Poisson (fréquence) × Log-Normal (sévérité)
    Dataset : freMTPL2 (677 991 polices RC automobile France)
    """
    multiplicateur = (
        RELATIVITES["Area"][area]          *
        RELATIVITES["DrivAge"][driv_age]   *
        RELATIVITES["VehAge"][veh_age]     *
        RELATIVITES["VehGas"][veh_gas]     *
        RELATIVITES["VehBrand"][veh_brand] *
        calculer_relativite_bm(bm)
    )
    prime_pure        = PRIME_PURE_REF * multiplicateur
    prime_commerciale = prime_pure / (1 - CHARGEMENT_TOTAL)
    return prime_pure, prime_commerciale, multiplicateur

def niveau_risque(ratio):
    """Détermine le niveau de risque selon le ratio vs référence"""
    if ratio < 0.85:
        return "🟢 Risque faible", "badge-faible", "#27ae60"
    elif ratio < 1.50:
        return "🟡 Risque modéré", "badge-moyen", "#f39c12"
    elif ratio < 5.00:
        return "🟠 Risque élevé", "badge-eleve", "#e67e22"
    else:
        return "🔴 Risque très élevé", "badge-tres-eleve", "#e74c3c"

# ============================================================
# SIDEBAR — NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:15px;'>
        <h2 style='color:white;'>🚗 TariAuto</h2>
        <p style='color:#bdc3c7; font-size:12px;'>
        Système de Tarification RC Automobile
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Accueil & Calculateur",
         "📊 Analyse du Portefeuille",
         "🔍 Comparaison de Profils",
         "📈 Sensibilité Tarifaire",
         "ℹ️ À propos du Modèle"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#bdc3c7; font-size:11px; text-align:center;'>
        <b>Données :</b> freMTPL2 (France)<br>
        677 991 polices analysées<br>
        <b>Modèle :</b> GLM Poisson × Log-Normal<br>
        <b>Auteur :</b> Thierry NIYOKWIZIGIRWA<br>
        <b>Version :</b> 2.0 — 2026
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 1 : ACCUEIL ET CALCULATEUR
# ============================================================
if page == "🏠 Accueil & Calculateur":

    # Titre
    st.markdown("""
    <div class='titre-principal'>
        <h1>🚗 TariAuto — Calculateur de Prime RC Automobile</h1>
        <p style='font-size:16px; opacity:0.9;'>
        Système de tarification basé sur un modèle actuariel 
        GLM Poisson × Log-Normal<br>
        Calibré sur 677 991 polices du marché français 
        (dataset freMTPL2)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Formulaire
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("📋 Saisie du profil assuré")

        with st.container():
            st.markdown("**👤 Conducteur**")
            driv_age = st.selectbox(
                "Classe d'âge du conducteur",
                list(RELATIVITES["DrivAge"].keys()),
                help="L'âge influence fortement la sinistralité"
            )
            bm = st.slider(
                "Coefficient Bonus-Malus",
                50, 230, 50, 1,
                help="50 = bonus maximum | > 100 = malus actif"
            )
            # Indicateur BM
            if bm == 50:
                st.success("✅ Bonus maximum — Excellent historique")
            elif bm <= 70:
                st.info("ℹ️ Bon historique de conduite")
            elif bm <= 100:
                st.warning("⚠️ Historique neutre")
            else:
                st.error(f"❌ Malus actif — {bm} points")

        st.markdown("---")

        with st.container():
            st.markdown("**🚘 Véhicule**")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                veh_age = st.selectbox(
                    "Âge du véhicule",
                    list(RELATIVITES["VehAge"].keys())
                )
                veh_gas = st.selectbox(
                    "Carburant",
                    list(RELATIVITES["VehGas"].keys())
                )
            with col_v2:
                veh_brand = st.selectbox(
                    "Marque",
                    list(RELATIVITES["VehBrand"].keys())
                )

        st.markdown("---")

        with st.container():
            st.markdown("**📍 Localisation**")
            area = st.selectbox(
                "Zone géographique",
                list(RELATIVITES["Area"].keys()),
                help="Zone A = rural → Zone F = métropole"
            )

        st.markdown("---")
        calculer = st.button("🧮 CALCULER MA PRIME", type="primary")

    with col_result:
        st.subheader("💰 Résultat tarifaire")

        # Calcul automatique (ou au clic)
        prime_pure, prime_commerciale, multiplicateur = calculer_prime(
            area, driv_age, veh_age, veh_gas, veh_brand, bm
        )

        prime_ref_commerciale = PRIME_PURE_REF / (1 - CHARGEMENT_TOTAL)
        ratio = prime_commerciale / prime_ref_commerciale
        niveau, badge_class, couleur_risque = niveau_risque(ratio)

        # Affichage du niveau de risque
        st.markdown(f"""
        <div style='text-align:center; margin-bottom:20px;'>
            <span class='{badge_class}' style='font-size:18px;'>
                {niveau}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Métriques principales
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Prime pure",
                f"{prime_pure:.2f} €",
                help="Coût technique du risque"
            )
        with c2:
            st.metric(
                "Prime commerciale",
                f"{prime_commerciale:.2f} €",
                f"{((ratio-1)*100):+.1f}% vs référence"
            )
        with c3:
            st.metric(
                "Multiplicateur",
                f"{multiplicateur:.3f}×",
                help="Produit des relativités"
            )

        st.markdown("---")

        # Décomposition graphique des relativités
        st.markdown("**📊 Décomposition des relativités**")

        facteurs = {
            "Zone"           : RELATIVITES["Area"][area],
            "Âge conducteur" : RELATIVITES["DrivAge"][driv_age],
            "BonusMalus"     : round(calculer_relativite_bm(bm), 4),
            "Âge véhicule"   : RELATIVITES["VehAge"][veh_age],
            "Carburant"      : RELATIVITES["VehGas"][veh_gas],
            "Marque"         : RELATIVITES["VehBrand"][veh_brand]
        }

        couleurs_barres = [
            "#27ae60" if v < 0.99
            else ("#e74c3c" if v > 1.01 else "#95a5a6")
            for v in facteurs.values()
        ]

        fig_rel = go.Figure(go.Bar(
            x=list(facteurs.keys()),
            y=list(facteurs.values()),
            marker_color=couleurs_barres,
            text=[f"{v:.4f}" for v in facteurs.values()],
            textposition="outside",
            textfont=dict(size=11)
        ))
        fig_rel.add_hline(
            y=1.0, line_dash="dash",
            line_color="black", line_width=1,
            annotation_text="Référence (1.000)",
            annotation_position="top right"
        )
        fig_rel.update_layout(
            height=280,
            margin=dict(t=20, b=10, l=10, r=10),
            showlegend=False,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#ecf0f1"),
            xaxis=dict(tickangle=-15)
        )
        st.plotly_chart(fig_rel, use_container_width=True)

        # Décomposition de la prime
        st.markdown("**🥧 Structure de la prime commerciale**")

        labels_pie = [
            "Risque technique (prime pure)",
            "Frais de gestion (25%)",
            "Marge de sécurité (10%)",
            "Réassurance (5%)"
        ]
        values_pie = [
            prime_pure,
            prime_commerciale * CHARGEMENTS["Frais de gestion"],
            prime_commerciale * CHARGEMENTS["Marge de sécurité"],
            prime_commerciale * CHARGEMENTS["Coût de réassurance"]
        ]

        fig_pie = go.Figure(go.Pie(
            labels=labels_pie,
            values=values_pie,
            hole=0.45,
            marker_colors=["#2980b9","#e74c3c","#f39c12","#27ae60"],
            textinfo="label+percent",
            textfont_size=10
        ))
        fig_pie.add_annotation(
            text=f"{prime_commerciale:.0f}€",
            x=0.5, y=0.5,
            font_size=18,
            font_color="#1e3a5f",
            showarrow=False
        )
        fig_pie.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================
# PAGE 2 : ANALYSE DU PORTEFEUILLE
# ============================================================
elif page == "📊 Analyse du Portefeuille":

    st.markdown("""
    <div class='titre-principal'>
        <h2>📊 Analyse du Portefeuille freMTPL2</h2>
        <p>677 991 polices RC automobile — Marché français</p>
    </div>
    """, unsafe_allow_html=True)

    # Statistiques clés
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Polices totales", "677 991")
    with col2:
        st.metric("Sinistres déclarés", "26 444")
    with col3:
        st.metric("Taux de sinistralité", "3.90%")
    with col4:
        st.metric("Prime pure moyenne", "80.59 €")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Distribution par zone géographique")
        zones = ["A","B","C","D","E","F"]
        effectifs = [103952, 75457, 191874, 151592, 137163, 17953]
        taux_sin = [0.0324, 0.0349, 0.0370, 0.0426, 0.0446, 0.0431]

        fig_zone = go.Figure()
        fig_zone.add_trace(go.Bar(
            name="Polices",
            x=zones, y=effectifs,
            marker_color="#2980b9",
            yaxis="y"
        ))
        fig_zone.add_trace(go.Scatter(
            name="Taux sinistralité",
            x=zones, y=taux_sin,
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=8),
            yaxis="y2"
        ))
        fig_zone.update_layout(
            height=350,
            yaxis=dict(title="Nombre de polices"),
            yaxis2=dict(
                title="Taux de sinistralité",
                overlaying="y", side="right",
                tickformat=".3f"
            ),
            legend=dict(x=0.01, y=0.99),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_zone, use_container_width=True)

    with col_g2:
        st.subheader("Distribution de la sinistralité")
        sinistres = [0,1,2,3,4,5,6,8,9,11,16]
        polices   = [653047,23571,1298,62,5,2,1,1,1,2,1]

        fig_sin = go.Figure(go.Bar(
            x=[str(s) for s in sinistres],
            y=polices,
            marker_color="#9b59b6",
            text=[f"{p:,}" for p in polices],
            textposition="outside"
        ))
        fig_sin.update_layout(
            height=350,
            xaxis_title="Nombre de sinistres",
            yaxis_title="Nombre de polices",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#ecf0f1")
        )
        st.plotly_chart(fig_sin, use_container_width=True)

    st.markdown("---")

    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.subheader("Taux de sinistralité par âge conducteur")
        ages  = ["18-25","26-35","36-45","46-55","56-65","65+"]
        taux  = [0.0347, 0.0307, 0.0399, 0.0430, 0.0382, 0.0389]
        colors = ["#e74c3c","#27ae60","#f39c12",
                  "#e74c3c","#f39c12","#f39c12"]

        fig_age = go.Figure(go.Bar(
            x=ages, y=taux,
            marker_color=colors,
            text=[f"{t:.4f}" for t in taux],
            textposition="outside"
        ))
        fig_age.add_hline(
            y=np.mean(taux), line_dash="dash",
            line_color="black",
            annotation_text=f"Moyenne : {np.mean(taux):.4f}"
        )
        fig_age.update_layout(
            height=350,
            xaxis_title="Classe d'âge",
            yaxis_title="Taux moyen",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#ecf0f1")
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_g4:
        st.subheader("Performance des modèles comparés")
        modeles = ["Poisson","Binomial Négatif","ZIP","XGBoost"]
        gini    = [0.3604, None, 0.3678, 0.3877]
        aic     = [150702, 150235, 149990, None]

        fig_mod = go.Figure()
        fig_mod.add_trace(go.Bar(
            name="Gini",
            x=modeles,
            y=[g if g else 0 for g in gini],
            marker_color="#2980b9",
            text=[f"{g:.4f}" if g else "N/A" for g in gini],
            textposition="outside"
        ))
        fig_mod.update_layout(
            height=350,
            yaxis_title="Coefficient de Gini",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#ecf0f1"),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_mod, use_container_width=True)

# ============================================================
# PAGE 3 : COMPARAISON DE PROFILS
# ============================================================
elif page == "🔍 Comparaison de Profils":

    st.markdown("""
    <div class='titre-principal'>
        <h2>🔍 Comparaison de Profils Tarifaires</h2>
        <p>Comparez jusqu'à 3 profils d'assurés simultanément</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 Renseignez les caractéristiques de 2 ou 3 profils "
            "pour les comparer")

    # Profils prédéfinis
    st.subheader("Profils types du portefeuille")

    profils_types = [
        {
            "nom"     : "Profil Référence",
            "area"    : "A — Zone rurale",
            "age"     : "18 — 25 ans",
            "veh_age" : "0 — 1 an (neuf)",
            "gas"     : "Diesel",
            "brand"   : "Marque Autres",
            "bm"      : 50,
            "couleur" : "#3498db"
        },
        {
            "nom"     : "Bon Profil",
            "area"    : "B — Zone semi-rurale",
            "age"     : "36 — 45 ans",
            "veh_age" : "2 — 5 ans",
            "gas"     : "Essence",
            "brand"   : "Marque B12",
            "bm"      : 50,
            "couleur" : "#27ae60"
        },
        {
            "nom"     : "Profil Moyen",
            "area"    : "C — Zone semi-urbaine",
            "age"     : "46 — 55 ans",
            "veh_age" : "6 — 10 ans",
            "gas"     : "Diesel",
            "brand"   : "Marque B2",
            "bm"      : 60,
            "couleur" : "#f39c12"
        },
        {
            "nom"     : "Profil Risqué",
            "area"    : "E — Zone très urbaine",
            "age"     : "18 — 25 ans",
            "veh_age" : "0 — 1 an (neuf)",
            "gas"     : "Essence",
            "brand"   : "Marque B3",
            "bm"      : 100,
            "couleur" : "#e74c3c"
        },
        {
            "nom"     : "Profil Très Risqué",
            "area"    : "F — Métropole",
            "age"     : "26 — 35 ans",
            "veh_age" : "0 — 1 an (neuf)",
            "gas"     : "Essence",
            "brand"   : "Marque B3",
            "bm"      : 150,
            "couleur" : "#8e44ad"
        }
    ]

    # Calcul des primes pour chaque profil
    resultats = []
    for p in profils_types:
        pp, pc, mult = calculer_prime(
            p["area"], p["age"], p["veh_age"],
            p["gas"], p["brand"], p["bm"]
        )
        resultats.append({
            "Profil"          : p["nom"],
            "Prime pure (€)"  : round(pp, 2),
            "Prime comm. (€)" : round(pc, 2),
            "Multiplicateur"  : round(mult, 3),
            "BM"              : p["bm"],
            "Zone"            : p["area"].split("—")[0].strip(),
            "couleur"         : p["couleur"]
        })

    # Graphique comparatif
    fig_comp = go.Figure()
    for r in resultats:
        fig_comp.add_trace(go.Bar(
            name=r["Profil"],
            x=[r["Profil"]],
            y=[r["Prime comm. (€)"]],
            marker_color=r["couleur"],
            text=[f"{r['Prime comm. (€)']:.0f} €"],
            textposition="outside",
            textfont=dict(size=12, color="black")
        ))

    fig_comp.update_layout(
        title="Primes commerciales par profil",
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        yaxis=dict(
            title="Prime commerciale (€)",
            gridcolor="#ecf0f1"
        ),
        bargap=0.3
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Tableau récapitulatif
    st.subheader("Tableau récapitulatif")
    df_resultats = pd.DataFrame(resultats).drop(columns=["couleur"])
    st.dataframe(
        df_resultats.style.highlight_max(
            subset=["Prime comm. (€)"],
            color="#fadbd8"
        ).highlight_min(
            subset=["Prime comm. (€)"],
            color="#d5f5e3"
        ),
        use_container_width=True
    )

    st.markdown("---")

    # Comparaison personnalisée
    st.subheader("🛠️ Créez votre comparaison personnalisée")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Profil A**")
        area_a    = st.selectbox("Zone A",
                    list(RELATIVITES["Area"].keys()), key="a1")
        age_a     = st.selectbox("Âge conducteur A",
                    list(RELATIVITES["DrivAge"].keys()), key="a2")
        bm_a      = st.slider("BonusMalus A", 50, 230, 50, key="a3")
        vehage_a  = st.selectbox("Âge véhicule A",
                    list(RELATIVITES["VehAge"].keys()), key="a4")
        gas_a     = st.selectbox("Carburant A",
                    list(RELATIVITES["VehGas"].keys()), key="a5")
        brand_a   = st.selectbox("Marque A",
                    list(RELATIVITES["VehBrand"].keys()), key="a6")

    with col_p2:
        st.markdown("**Profil B**")
        area_b    = st.selectbox("Zone B",
                    list(RELATIVITES["Area"].keys()), key="b1",
                    index=4)
        age_b     = st.selectbox("Âge conducteur B",
                    list(RELATIVITES["DrivAge"].keys()), key="b2",
                    index=0)
        bm_b      = st.slider("BonusMalus B", 50, 230, 120, key="b3")
        vehage_b  = st.selectbox("Âge véhicule B",
                    list(RELATIVITES["VehAge"].keys()), key="b4",
                    index=0)
        gas_b     = st.selectbox("Carburant B",
                    list(RELATIVITES["VehGas"].keys()), key="b5",
                    index=1)
        brand_b   = st.selectbox("Marque B",
                    list(RELATIVITES["VehBrand"].keys()), key="b6",
                    index=2)

    _, pc_a, _ = calculer_prime(
        area_a, age_a, vehage_a, gas_a, brand_a, bm_a)
    _, pc_b, _ = calculer_prime(
        area_b, age_b, vehage_b, gas_b, brand_b, bm_b)

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Prime Profil A", f"{pc_a:.2f} €")
    with col_r2:
        st.metric("Prime Profil B", f"{pc_b:.2f} €")
    with col_r3:
        diff = pc_b - pc_a
        st.metric(
            "Différence A → B",
            f"{abs(diff):.2f} €",
            f"{'+' if diff>0 else ''}{diff:.2f} €"
        )

# ============================================================
# PAGE 4 : SENSIBILITE TARIFAIRE
# ============================================================
elif page == "📈 Sensibilité Tarifaire":

    st.markdown("""
    <div class='titre-principal'>
        <h2>📈 Analyse de Sensibilité Tarifaire</h2>
        <p>Impact de chaque facteur sur la prime commerciale</p>
    </div>
    """, unsafe_allow_html=True)

    # Profil de base pour la sensibilité
    st.subheader("Profil de base pour l'analyse")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        base_area  = st.selectbox("Zone de base",
                     list(RELATIVITES["Area"].keys()), index=2)
        base_age   = st.selectbox("Âge de base",
                     list(RELATIVITES["DrivAge"].keys()), index=2)
    with col_s2:
        base_vehage = st.selectbox("Âge véhicule base",
                      list(RELATIVITES["VehAge"].keys()), index=2)
        base_gas    = st.selectbox("Carburant base",
                      list(RELATIVITES["VehGas"].keys()))
    with col_s3:
        base_brand = st.selectbox("Marque base",
                     list(RELATIVITES["VehBrand"].keys()))
        base_bm    = st.slider("BonusMalus base", 50, 230, 60)

    _, prime_base, _ = calculer_prime(
        base_area, base_age, base_vehage,
        base_gas, base_brand, base_bm
    )
    st.info(f"**Prime commerciale de base : {prime_base:.2f} €**")

    st.markdown("---")

    # Sensibilité au BonusMalus
    col_sens1, col_sens2 = st.columns(2)

    with col_sens1:
        st.subheader("Impact du Bonus-Malus")
        bm_range    = list(range(50, 231, 5))
        primes_bm   = []
        for bm_val in bm_range:
            _, pc, _ = calculer_prime(
                base_area, base_age, base_vehage,
                base_gas, base_brand, bm_val
            )
            primes_bm.append(round(pc, 2))

        fig_bm = go.Figure()
        fig_bm.add_trace(go.Scatter(
            x=bm_range, y=primes_bm,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(41, 128, 185, 0.1)",
            line=dict(color="#2980b9", width=2),
            name="Prime"
        ))
        fig_bm.add_vline(
            x=base_bm, line_dash="dash",
            line_color="red",
            annotation_text=f"Votre BM={base_bm}"
        )
        fig_bm.update_layout(
            height=300,
            xaxis_title="Coefficient BonusMalus",
            yaxis_title="Prime (€)",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#ecf0f1"),
            showlegend=False
        )
        st.plotly_chart(fig_bm, use_container_width=True)

    with col_sens2:
        st.subheader("Impact de la zone géographique")
        zones_labels = list(RELATIVITES["Area"].keys())
        primes_zones = []
        for z in zones_labels:
            _, pc, _ = calculer_prime(
                z, base_age, base_vehage,
                base_gas, base_brand, base_bm
            )
            primes_zones.append(round(pc, 2))

        couleurs_zones = [
            "#27ae60" if pc <= prime_base
            else "#e74c3c"
            for pc in primes_zones
        ]

        fig_zones = go.Figure(go.Bar(
            x=[z.split("—")[1].strip() for z in zones_labels],
            y=primes_zones,
            marker_color=couleurs_zones,
            text=[f"{p:.0f}€" for p in primes_zones],
            textposition="outside"
        ))
        fig_zones.add_hline(
            y=prime_base, line_dash="dash",
            line_color="black",
            annotation_text=f"Base: {prime_base:.0f}€"
        )
        fig_zones.update_layout(
            height=300,
            xaxis_title="Zone",
            yaxis_title="Prime (€)",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#ecf0f1"),
            showlegend=False
        )
        st.plotly_chart(fig_zones, use_container_width=True)

    st.markdown("---")

    # Stress tests
    st.subheader("🔥 Stress-tests — Scénarios adverses")

    scenarios = {
        "Sinistralité +10%"     : prime_base * 1.10,
        "Sinistralité +20%"     : prime_base * 1.20,
        "Réparations +20%"      : prime_base * 1.20,
        "Inflation médicale +15%": prime_base * 1.15,
        "Charg. sécurité 15%"   : (prime_base *
                                    (1-CHARGEMENT_TOTAL) /
                                    (1-0.40+0.05)),
        "Référence"              : prime_base,
        "Sinistralité -10%"     : prime_base * 0.90,
    }

    fig_stress = go.Figure(go.Bar(
        x=list(scenarios.keys()),
        y=list(scenarios.values()),
        marker_color=[
            "#e74c3c" if v > prime_base
            else ("#27ae60" if v < prime_base else "#3498db")
            for v in scenarios.values()
        ],
        text=[f"{v:.2f}€" for v in scenarios.values()],
        textposition="outside"
    ))
    fig_stress.add_hline(
        y=prime_base, line_dash="dash",
        line_color="black",
        annotation_text=f"Prime de base: {prime_base:.2f}€"
    )
    fig_stress.update_layout(
        height=380,
        xaxis_title="Scénario",
        yaxis_title="Prime commerciale (€)",
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#ecf0f1"),
        showlegend=False,
        xaxis=dict(tickangle=-20)
    )
    st.plotly_chart(fig_stress, use_container_width=True)

# ============================================================
# PAGE 5 : A PROPOS DU MODELE
# ============================================================
elif page == "ℹ️ À propos du Modèle":

    st.markdown("""
    <div class='titre-principal'>
        <h2>ℹ️ À propos du Modèle Actuariel</h2>
        <p>Documentation technique de TariAuto</p>
    </div>
    """, unsafe_allow_html=True)

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.subheader("📐 Modèle statistique")
        st.markdown("""
        **Approche Fréquence × Sévérité**
        
        La prime pure est calculée comme :
        
        $$\\hat{S}_i = \\hat{\\lambda}_i \\times \\hat{\\mu}_i 
        \\times \\tau$$
        
        Où :
        - $\\hat{\\lambda}_i$ = fréquence prédite (GLM Poisson)
        - $\\hat{\\mu}_i$ = sévérité prédite (Log-Normal)
        - $\\tau = 1.061$ = facteur multi-sinistres
        
        **Modèle de fréquence (GLM Poisson)**
        
        $$\\log(\\lambda_i) = \\beta_0 + \\sum_j \\beta_j x_{ji} 
        + \\log(E_i)$$
        
        **Métriques de performance :**
        | Modèle | AIC | Gini |
        |--------|-----|------|
        | Poisson | 150 702 | 0.3604 |
        | Bin. Négatif | 150 235 | — |
        | ZIP | 149 990 | 0.3678 |
        | **XGBoost** | — | **0.3877** |
        """)

    with col_info2:
        st.subheader("🗄️ Données utilisées")
        st.markdown("""
        **Dataset freMTPL2 (CASdatasets, R)**
        
        | Indicateur | Valeur |
        |-----------|--------|
        | Polices analysées | 677 991 |
        | Sinistres observés | 26 444 |
        | Taux de sinistralité | 3.90% |
        | Période | ~2011-2013 |
        | Marché | RC Auto France |
        
        **Variables explicatives retenues :**
        - `Area` : zone géographique (A→F)
        - `DrivAge` : âge conducteur (6 classes)
        - `VehAge` : âge véhicule (6 classes)
        - `VehGas` : type carburant (2 modalités)
        - `VehBrand` : marque véhicule (9 modalités)
        - `BonusMalus` : coefficient [50, 230]
        
        **Structure tarifaire :**
        
        $$\\Pi = \\frac{S_{pure}}{1 - (c_g + c_s + c_r)}$$
        
        Avec $c_g=25\\%$, $c_s=10\\%$, $c_r=5\\%$ 
        → **Chargement total = 40%**
        """)

    st.markdown("---")

    st.subheader("🏗️ Architecture technique")
    col_tech1, col_tech2, col_tech3 = st.columns(3)

    with col_tech1:
        st.markdown("""
        **Backend / Analyse**
        - 🔵 Python 3.10+
        - 📊 Pandas, NumPy
        - 📈 Scikit-learn
        - 🔢 R 4.3.2 (modélisation)
        """)

    with col_tech2:
        st.markdown("""
        **Frontend / Visualisation**
        - 🌐 Streamlit 1.x
        - 📉 Plotly
        - 🎨 CSS personnalisé
        - 📱 Design responsive
        """)

    with col_tech3:
        st.markdown("""
        **Référence scientifique**
        - Ohlsson & Johansson (2010)
        - Noll, Salzmann & Wüthrich (2020)
        - Boucher et al. (2007)
        - Lambert (1992) — modèle ZIP
        """)

    st.markdown("---")

    # Déclaration
    st.info("""
    **📋 Déclaration**
    
    Cette application a été développée dans le cadre du Master 2 
    Actuariat et Finance, Université du Burundi (2025-2026).
    Elle est basée sur une analyse actuarielle rigoureuse du 
    dataset freMTPL2 comprenant 677 991 polices d'assurance RC 
    automobile du marché français.
    
    **Auteur :** Thierry NIYOKWIZIGIRWA  
    **Encadrant :** Dr. William SAHINGUVU
    **Date :** 19 Septembre, 2026
    """)

# ============================================================
# FOOTER GLOBAL
# ============================================================
st.markdown("""
<div class='footer'>
    <b>TariAuto v2.0</b> — Système de Tarification RC Automobile<br>
    Master 2 Actuariat et Finance — Université du Burundi — 2026<br>
    <small>Basé sur freMTPL2 (677 991 polices) | 
    GLM Poisson × Log-Normal | Python + Streamlit</small>
</div>
""", unsafe_allow_html=True)
