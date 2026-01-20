import streamlit as st

import xarray as xr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 1. CONFIGURATION ET TITRE
st.set_page_config(layout="wide", page_title="Climat 1950-1984-Laignel Margot et Hajmi Hanaa")

st.title("Visualiser l'accélération du réchauffement climatique en France métropolitaine de 1950 à 1984")
st.subheader("Problématique : Comment le réchauffement climatique se manifeste-t-il en France métropolitaine entre 1950 et 1984, depuis les tendances moyennes globales jusqu’aux variations saisonnières")

st.markdown("""
Cette application a pour but d'analyser l'évolution du climat sur 35 ans.
Nous allons partir d'une vision globale (les moyennes) pour descendre vers le détail (les saisons, les jours précis, et la carte).
""")

# 2. CHARGEMENT DES DONNÉES
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


# 3. LES ONGLETS DE VISUALISATION

if df_france is not None:

    # Création des onglets
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "0. Tableau de Bord",
        "1. Moyennes Annuelles",
        "2. Cycles Saisonniers",
        "3. Températures Réelles",
        "4. Focus 1976",
        "5. Décennies",
        "6. Comparateur Périodes",
        "7. Carte de France"
    ])

# ONGLET 0 : TABLEAU DE BORD (VUE GÉNÉRALE)

    with tab0:
     st.title("Tableau de Bord Climatique (1950-1984)")
     st.markdown("---")

     st.write("### Chiffres Clés : France Métropolitaine (1950-1984)")

    # 1. Calcul des Chiffres Clés (KPIs)
     moyenne_globale = df_france[col_temp].mean()

    # Record de chaud
     rec_chaud = df_france.loc[df_france[col_temp].idxmax()]
     temp_max = rec_chaud[col_temp]
     date_max = rec_chaud['time'].strftime('%d %B %Y')

     # Record de froid
     rec_froid = df_france.loc[df_france[col_temp].idxmin()]
     temp_min = rec_froid[col_temp]
     date_min = rec_froid['time'].strftime('%d %B %Y')

    # 2. Affichage en 3 colonnes (Grosses Métriques)
     col1, col2, col3 = st.columns(3)

     with col1:
        st.metric(label="Moyenne sur 35 ans", value=f"{moyenne_globale:.1f} °C")

     with col2:
        st.metric(label="Record de Chaleur", value=f"{temp_max:.1f} °C", delta=date_max)

     with col3:
        st.metric(label="Record de Froid", value=f"{temp_min:.1f} °C", delta=date_min, delta_color="inverse")

    st.markdown("---")


    # ONGLET 1 : ANOMALIE ANNUELLE
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

        st.info("**Apport à la problématique :** Entre 1950 et 1985, les anomalies de température annuelle en France présentent une forte variabilité interannuelle, sans tendance linéaire marquée. Plusieurs épisodes contrastés alternent entre anomalies positives et négatives, traduisant une dominance de la variabilité naturelle du climat sur cette période.")


    # ONGLET 2 : SAISONS (BOXPLOT)
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
        st.header("Matrice Thermique : Saisons vs Décennies")

# 1. Préparation des données
        df_visu = df_france.copy()

# Création de la colonne Saison
        df_visu["Saison"] = df_visu["Month"].map({
    12: "Hiver", 1: "Hiver", 2: "Hiver",
    3: "Printemps", 4: "Printemps", 5: "Printemps",
    6: "Été", 7: "Été", 8: "Été",
    9: "Automne", 10: "Automne", 11: "Automne"
})

# Création de la colonne Décennie (1950, 1960, 1970...)
        df_visu['Decennie'] = (df_visu['Year'] // 10) * 10

# 2. Calcul de la MOYENNE
        df_moyenne = df_visu.groupby(['Decennie', 'Saison'])[col_temp].mean().reset_index()

# 3. Création du graphique
        fig = px.density_heatmap(
          df_moyenne,
          x="Decennie",
          y="Saison",
          z=col_temp,
          histfunc="avg", # Moyenne
          color_continuous_scale="RdBu_r", # Bleu (froid) -> Rouge (chaud)
          title="Température moyenne par Décennie et Saison",
          text_auto=".2f", # Affiche la valeur avec 2 décimales
          category_orders={"Saison": ["Hiver", "Printemps", "Été", "Automne"]} # Ordre logique
    )

# Ajustement de la taille et des axes pour faire propre
        fig.update_layout(
            xaxis_title="Décennie",
            yaxis_title="Saison",
            xaxis=dict(type='category'),
            height=500
        )

# Affichage Streamlit
        st.plotly_chart(fig, use_container_width=True)

# 4. Description et Analyse (Demandé)
        st.markdown("### Description")
        st.write("""
                 Ici, j'ai croisé les saisons et les décennies. Le truc le plus flagrant, c'est la ligne "Été". Au début (années 50), la case est orange clair/rouge pâle.
                 Mais plus on va vers la droite (années 80), plus la case devient rouge foncé.
                  Les autres saisons (Hiver, Automne) ne changent pas autant, elles restent assez stables
                  L'échelle de couleur permet d'identifier immédiatement les anomalies : le **bleu** pour le froid et le **rouge** pour le chaud.
                 """)

        st.info("""
          **Apport à la problématique :**
             Ça prouve que le réchauffement ne touche pas toute l'année de la même façon. Pour l'instant,
                ce sont surtout nos étés qui deviennent plus intenses et plus chauds.
                C'est le premier signe visible du changement climatique en France.**.
                  """)

    # ONGLET 3 : TEMPÉRATURES RÉELLES (SCATTER)
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

    # ONGLET 4 : FOCUS 1976
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
        st.write("La courbe grise montre la moyenne habituelle. " \
        "La courbe rouge, c'est l'année 1976. On voit qu'en été, la courbe rouge s'envole complètement au-dessus de la normale. " \
        "C'est un pic énorme qui dure plusieurs mois")

        st.info("**Apport à la problématique :** Les moyennes lissent tout, mais la réalité, c'est aussi des chocs. " \
        "Le réchauffement climatique, ce n'est pas juste 'il fait un peu plus doux', " \
        "c'est l'apparition d'années 'catastrophes' comme 1976 (sécheresse) qui sortent complètement de la norme.")


    # ONGLET 5 : Décennies (BOXPLOT)
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


    # ONGLET 6 : COMPARATEUR GÉNÉRAL (VIOLIN + STATS)
    with tab6:
        st.header("6. Comparateur Universel")
        st.write("Cet outil analyse la 'signature thermique' de la période choisie et la compare mathématiquement au reste de l'histoire.")

        st.markdown("---")

        # 1. Le Sélecteur de Période (Double Slider)
        annee_min = int(df_france['Year'].min())
        annee_max = int(df_france['Year'].max())

        col_sel, col_kpi = st.columns([2, 1])

        with col_sel:
            periode = st.slider(
                "Définir la période cible (Rouge) :",
                min_value=annee_min,
                max_value=annee_max,
                value=(1950, 1960) # Valeur par défaut
            )
            nom_periode = f"Période {periode[0]}-{periode[1]}"
            nom_reste = "Reste de l'historique"

        # 2. Préparation des données
        df_comp = df_france.copy()

        # On étiquette les données : "Cible" vs "Reste"
        # Astuce : On utilise une fonction simple pour créer la colonne 'Groupe'
        df_comp['Groupe'] = df_comp['Year'].apply(
            lambda x: nom_periode if periode[0] <= x <= periode[1] else nom_reste
        )

        # 3. Calcul des Moyennes pour la comparaison chiffrée
        moy_periode = df_comp[df_comp['Groupe'] == nom_periode][col_temp].mean()
        moy_reste = df_comp[df_comp['Groupe'] == nom_reste][col_temp].mean()
        delta = moy_periode - moy_reste

        # Affichage du KPI dans la colonne de droite
        with col_kpi:
            st.metric(
                label=f"Moyenne {nom_periode}",
                value=f"{moy_periode:.2f} °C",
                delta=f"{delta:+.2f} °C vs Reste",
                delta_color="normal"
            )

        # 4. Le Graphique "Violin" (Vision Générale)
        fig_violin = px.violin(
            df_comp,
            y=col_temp,       # Température en vertical
            x="Groupe",       # Les deux groupes côte à côte
            color="Groupe",   # Couleur distincte
            box=True,         # Affiche la médiane (la barre au milieu du violon)
            points=False,
            title=f"Signature Thermique : {nom_periode} vs Reste",
            color_discrete_map={nom_reste: "lightgrey", nom_periode: "#E74C3C"}
        )

        fig_violin.update_layout(
            yaxis_title="Température (°C)",
            xaxis_title="",
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig_violin, use_container_width=True)

        # 5. Interprétation Automatique
        st.info(f"""
        **Analyse rapide :**
        * La moyenne de votre sélection est de **{moy_periode:.1f}°C**.
        * Si le violon rouge est situé globalement **plus haut** que le gris, cela confirme un réchauffement sur cette période.
        * La largeur du violon indique la fréquence : une forme plus "ventrue" vers le haut signifie des jours chauds plus fréquents.
        """)

    # ONGLET 7 : CARTE ANIMÉE
    with tab7:
        st.header("Animation thermique de la France")

        # Préparation (Moyenne annuelle pour l'animation)
        da_temp = ds_raw[col_temp]
        ds_map_viz = da_temp.resample(time='1AS').mean(skipna=True)

        fig5 = px.imshow(
            ds_map_viz,
            animation_frame="time",
            origin='lower',
            aspect='equal',
            color_continuous_scale="RdBu_r",
            range_color=[-5, 25],
            title="Animation thermique de la France (Annuel)",
            labels={'color': 'Température (°C)', 'latitude': 'Latitude', 'longitude': 'Longitude'}
        )

        fig5.layout.sliders[0].currentvalue.prefix = "Date : "
        for step in fig5.layout.sliders[0].steps:
            step["label"] = pd.to_datetime(step["label"]).strftime('%Y')

        fig5.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("**Description :**")
        st.write("Une carte animée mois par mois qui montre l'évolution des températures sur le territoire.")

        st.info("**Apport à la problématique :** Le réchauffement interagit avec la géographie locale. " \
        "Il ne s'agit pas d'une augmentation uniforme de +1°C partout. " \
        "Les 'réalités locales' montrent que certains territoires sont des tampons thermiques (littoral) tandis que d'autres sont des amplificateurs (vallées intérieures), " \
        "modulant la façon dont le réchauffement est vécu localement.")
else:
    st.error("Erreur : Impossible de charger le fichier de données. Vérifie qu'il est bien présent.")