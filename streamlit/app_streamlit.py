import streamlit as st
import xarray as xr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. CONFIGURATION ET TITRE
# ==========================================
st.set_page_config(layout="wide", page_title="Climat 1950-1984")

st.title("Visualiser l'accélération du réchauffement climatique en France métropolitaine de 1950 à 1984")
st.subheader("Problématique : Comment le réchauffement climatique se manifeste-t-il en France métropolitaine entre 1950 et 1984, depuis les tendances moyennes globales jusqu’aux variations saisonnières")

st.markdown("""
Cette application a pour but d'analyser l'évolution du climat sur 35 ans.
Nous allons partir d'une vision globale (les moyennes) pour descendre vers le détail (les saisons, les jours précis, et la carte).
""")

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_resource
def load_data():
    # Gestion du chemin de fichier
    file_path = Path("..") / "donnees_carte_35ans_journalier.nc"
    if not file_path.exists():
        file_path = Path("donnees_carte_35ans_journalier.nc")

    try:
        ds = xr.open_dataset(file_path)

        # Moyenne spatiale pour les graphiques 2D
        ds_mean = ds.mean(dim=['latitude', 'longitude'], skipna=True)
        df = ds_mean.to_dataframe().reset_index()

        # Nettoyage et renommage
        if 'valid_time' in df.columns:
            df = df.rename(columns={'valid_time': 'time'})

        col_temp = 'Temperature_C' if 'Temperature_C' in df.columns else 't2m'

        # Création des colonnes temporelles
        df['time'] = pd.to_datetime(df['time'])
        df['Year'] = df['time'].dt.year
        df['Month'] = df['time'].dt.month
        df['DayOfYear'] = df['time'].dt.dayofyear

        # Ajout de la colonne Saison
        season_map = {
            1: 'Hiver', 2: 'Hiver', 12: 'Hiver',
            3: 'Printemps', 4: 'Printemps', 5: 'Printemps',
            6: 'Été', 7: 'Été', 8: 'Été',
            9: 'Automne', 10: 'Automne', 11: 'Automne'
        }
        df['Saison'] = df['Month'].map(season_map)

        return df, ds, col_temp

    except Exception as e:
        return None, None, None

df_france, ds_raw, col_temp = load_data()

# ==========================================
# 3. LES ONGLETS DE VISUALISATION
# ==========================================

if df_france is not None:

    # Création des onglets
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Moyennes Annuelles",
        "2. Cycles Saisonniers",
        "3. Températures Réelles",
        "4. Focus 1976",
        "5. Décennies",
        "6. Carte de France"
    ])

    # -------------------------------------------------------
    # ONGLET 1 : ANOMALIE ANNUELLE
    # -------------------------------------------------------
    with tab1:
        st.header("Anomalie de température annuelle")

        # 1. Agrégation par année
        df_yearly = df_france.groupby('Year')[col_temp].mean().reset_index()
        # 2. Moyenne de référence
        ref_mean = df_yearly[col_temp].mean()
        # 3. Calcul anomalie
        df_yearly['Anomalie'] = df_yearly[col_temp] - ref_mean

        # 4. Graphique
        fig1 = px.line(
            df_yearly,
            x='Year',
            y='Anomalie',
            markers=True,
            title="Anomalie de température annuelle (France, 35 ans)",
            labels={'Anomalie': 'Écart à la moyenne (°C)', 'Year': 'Année'}
        )
        fig1.add_hline(y=0, line_dash="dash", line_color="grey")

        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("**Description :**")
        st.write("Ce graphique montre l'écart de température de chaque année par rapport à la moyenne globale de la période 1950-1985. La ligne pointillée grise représente la moyenne.")

        st.info("**Apport à la problématique :** Ce graphique répond à la première partie du sujet : les moyennes annuelles. Il permet de voir s'il y a une tendance de fond au réchauffement ou si le climat alterne simplement entre années chaudes et froides.")

    # -------------------------------------------------------
    # ONGLET 2 : SAISONS (BOXPLOT)
    # -------------------------------------------------------
    with tab2:
        st.header("Distribution des températures par Saison")

        # Calcul des min/max globaux
        min_y = df_france[col_temp].min() - 5
        max_y = df_france[col_temp].max() + 5

        fig2 = px.box(
            df_france.sort_values('Year'),
            x="Saison",
            y=col_temp,
            animation_frame="Year",
            color="Saison",
            category_orders={"Saison": ["Hiver", "Printemps", "Été", "Automne"]},
            range_y=[min_y, max_y],
            title="Distribution des températures par Saison (Évolution annuelle)",
            labels={col_temp: "Température journalière (°C)"}
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Description :**")
        st.write("Ce diagramme en boîte montre la dispersion des températures pour chaque saison. L'animation permet de faire défiler les années une par une.")

        st.info("**Apport à la problématique :** Il permet d'affiner l'analyse annuelle. On cherche à voir si le changement climatique affecte toutes les saisons de la même manière ou si ce sont seulement les étés qui deviennent plus chauds et variables.")

    # -------------------------------------------------------
    # ONGLET 3 : TEMPÉRATURES RÉELLES (SCATTER)
    # -------------------------------------------------------
    with tab3:
        st.header("Températures réelles jour par jour")

        # Préparation
        df_visu = df_france.sort_values(by=['Year', 'DayOfYear']).copy()
        y_min = df_visu[col_temp].min() - 2
        y_max = df_visu[col_temp].max() + 2

        fig3 = px.scatter(
            df_visu,
            x="DayOfYear",
            y=col_temp,
            animation_frame="Year",
            animation_group="DayOfYear",
            color=col_temp,
            color_continuous_scale="RdBu_r",
            range_y=[y_min, y_max],
            title="Températures réelles jour par jour",
            hover_data={'DayOfYear': True, col_temp: ':.1f'}
        )

        fig3.update_traces(marker=dict(size=5), mode='markers+lines')
        fig3.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
        fig3.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, annotation_text="Seuil de Gel (0°C)")
        fig3.update_layout(xaxis_title="Jour de l'année", yaxis_title="Température (°C)")

        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Description :**")
        st.write("Chaque point représente la température moyenne de la France pour un jour donné. Le curseur permet de naviguer dans le temps pour observer la courbe annuelle réelle.")

        st.info("**Apport à la problématique :** On quitte les moyennes lissées pour voir la réalité quotidienne. Cela permet de repérer les vagues de froid (points bleus sous la ligne de gel) et les vagues de chaleur invisibles sur une simple moyenne annuelle.")

    # -------------------------------------------------------
    # ONGLET 4 : FOCUS 1976
    # -------------------------------------------------------
    with tab4:
        st.header("Comparaison : L'année 1976 vs la Normale")

        # Calculs
        df_climat = df_france.groupby('DayOfYear')[col_temp].mean().reset_index()
        df_1976 = df_france[df_france['Year'] == 1976]

        fig4 = go.Figure()
        # Moyenne
        fig4.add_trace(go.Scatter(
            x=df_climat['DayOfYear'], y=df_climat[col_temp],
            mode='lines', name='Moyenne 1950-1985',
            line=dict(color='grey', width=1),
            fill='tozeroy', fillcolor='rgba(200,200,200,0.2)'
        ))
        # 1976
        fig4.add_trace(go.Scatter(
            x=df_1976['DayOfYear'], y=df_1976[col_temp],
            mode='lines', name='Année 1976',
            line=dict(color='red', width=2)
        ))

        fig4.update_layout(
            title="L'anomalie de 1976 comparée à la normale",
            xaxis_title="Jour de l'année", yaxis_title="Température (°C)"
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**Description :**")
        st.write("La courbe grise représente la moyenne des températures sur 35 ans. La courbe rouge représente uniquement l'année 1976.")

        st.info("**Apport à la problématique :** Ce graphique illustre la notion d'événement extrême. Il montre concrètement ce que signifie une année 'anormale' par rapport à la moyenne, avec ici une sécheresse et une canicule estivale très marquée.")

   # -------------------------------------------------------
    # ONGLET 5 : Décennies (BOXPLOT)
    # -------------------------------------------------------

    with tab5:
        st.header("Évolution par Décennie")

        # 1. Création de la colonne "Décennie" (1950s, 1960s...)
        df_france['Decennie'] = (df_france['Year'] // 10) * 10
        df_france['Decennie_Label'] = df_france['Decennie'].astype(str) + "s"

        # 2. Sélecteur pour filtrer (Astuce pour mieux voir)
        st.write("Le réchauffement ne se voit pas forcément sur toute l'année mélangée. Choisissez un mois (ex: Juillet) pour voir si la température décennale augmente.")

        mois_liste = {1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
                      7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'}

        choix_mois = st.selectbox("Choisir le mois à analyser :", list(mois_liste.values()), index=6) # Juillet par défaut

        # On retrouve le numéro du mois
        num_mois = [k for k, v in mois_liste.items() if v == choix_mois][0]

        # 3. Filtrage des données
        df_decennie = df_france[df_france['Month'] == num_mois].sort_values('Year')

        # 4. Le Boxplot par tranche
        fig_dec = px.box(
            df_decennie,
            x="Decennie_Label",
            y=col_temp,
            color="Decennie_Label",
            title=f"Distribution des températures de {choix_mois} par décennie",
            labels={col_temp: "Température (°C)", "Decennie_Label": "Période"},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        fig_dec.update_layout(showlegend=False) # Pas besoin de légende car c'est écrit en bas
        st.plotly_chart(fig_dec, use_container_width=True)

        st.markdown("**Description :**")
        st.write("Ce graphique regroupe les températures par tranches de 10 ans. La ligne au milieu de la boîte représente la médiane. Si les boîtes se décalent vers le haut de gauche à droite, c'est le signe d'un réchauffement.")

        st.info("**Apport à la problématique :** Cela permet de valider l'accélération du phénomène. On cherche à vérifier si la 'normale' des années 80 est visiblement plus élevée que celle des années 50, en isolant les variations saisonnières.")



    # -------------------------------------------------------
    # ONGLET 6 : CARTE ANIMÉE
    # -------------------------------------------------------
    with tab6:
        st.header("Animation thermique de la France")

        # Préparation (Moyenne mensuelle pour l'animation)
        da_temp = ds_raw[col_temp]
        ds_map_viz = da_temp.resample(time='1MS').mean(skipna=True)

        fig5 = px.imshow(
            ds_map_viz,
            animation_frame="time",
            origin='lower',
            aspect='equal',
            color_continuous_scale="RdBu_r",
            range_color=[-5, 25],
            title="Animation thermique de la France (Mensuel)",
            labels={'color': 'Température (°C)', 'latitude': 'Latitude', 'longitude': 'Longitude'}
        )

        fig5.layout.sliders[0].currentvalue.prefix = "Date : "
        # Formatage des dates du slider
        for step in fig5.layout.sliders[0].steps:
            step["label"] = pd.to_datetime(step["label"]).strftime('%Y-%m')

        # On enlève les axes numériques
        fig5.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("**Description :**")
        st.write("Une carte animée mois par mois qui montre l'évolution des températures sur le territoire.")
        st.info("**Apport à la problématique :** C'est la partie 'réalités locales' du sujet. Cela permet de voir que le climat n'est pas uniforme : les zones montagneuses et littorales réagissent différemment aux changements de température.")
else:
    st.error("Erreur : Impossible de charger le fichier de données. Vérifie qu'il est bien présent.")