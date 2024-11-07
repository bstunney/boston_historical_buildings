import pandas as pd
from geopy.geocoders import Nominatim
from datetime import datetime
import matplotlib.pyplot as plt
import geojson
import geopandas as gpd
from shapely.geometry import Point, Polygon
import seaborn as sns

from scipy import spatial
import numpy as np
import math

def vis(geo_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    # ax.set_xlim(min(list(wdf['lon'])), max(list(wdf['lon'])))
    # ax.set_ylim(min(list(wdf['lat'])), max(list(wdf['lat'])))

    street_map = gpd.read_file(
        "/Users/bentunney/Desktop/courses/INSH 2102/bography final project/Boston_Neighborhoods_StatAreas 2017/Boston_Neighborhoods.shp")

    street_map.to_crs(epsg=4326).plot(ax=ax, color='lightgrey')
    # street_map.plot(ax=ax, alpha=0.4, color='lightgrey') , alpha=0.4,color='lightgrey')

    geo_df.plot(column="rel", ax=ax, cmap="tab10", legend=True,
                alpha=0.75,
                markersize=20,
                marker='o')
    """
    geo_df.plot( column = "change_rel", ax=ax,cmap = "rainbow", legend = True,
               alpha= 0.8,
               markersize=20,
               marker='o')
    """

    plt.text(-71.1265, 42.2832, 'Roslindale', fontsize=5)
    plt.text(-71.0899, 42.3126, 'Roxbury', fontsize=5)
    plt.text(-71.1141, 42.3132, 'Jamaica Plain', fontsize=5)
    plt.text(-71.0935, 42.2772, 'Mattapan', fontsize=5)
    plt.text(-71.1060, 42.3299, 'Mission Hill', fontsize=5)
    plt.text(-71.0649, 42.2995, 'Dorchester', fontsize=5)
    plt.text(-71.1256, 42.2557, 'Hyde Park', fontsize=5)
    plt.text(-71.1498, 42.3472, 'Allston', fontsize=5)

    plt.title("Boston Worship Places vs. Religious Group")

    # [geo_df['lon'] < 43]

    plt.show()

def get_wdf(idf):
    wdf = idf[idf["building_typology"] == "Worship"]
    wdf["change_rel"] = ['N', 'N', 'N', 'N', 'N', 'D', 'N', 'N', 'D', 'A',
                         'N', 'S', 'D', 'A', 'D', 'N', 'A', 'N', 'N', 'A',
                         'N', 'N', 'N', 'N', 'N', 'A', 'N', 'A', 'N', 'N',
                         'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
                         'N', 'N', 'N', 'A', 'N', 'N', 'N']

    wdf = wdf[wdf["change_rel"] != "A"]

    wdf.loc[wdf['change_rel'] == 'N', 'change_rel'] = "No Religion Previously"
    wdf.loc[wdf['change_rel'] == 'S', 'change_rel'] = "Same Religion Previously"
    wdf.loc[wdf['change_rel'] == 'D', 'change_rel'] = "Different Religion Previously"

    wdf["rel"] = ["Cao Dai", "Christian", "Christian", "Christian", "Christian",
                  "Christian", "Christian", "Christian", "Christian", "Christian",
                  "Christian", "Christian", "Christian", "Christian", "Christian",
                  "Christian", "Other", "Christian", "Christian", "Christian",
                  "Christian", "Christian", "Christian", "Christian", "Christian",
                  "Christian", "Christian", "Christian", "Christian", "Christian",
                  "Christian", "Christian", "Christian", "Christian", "Christian",
                  "Christian", "Christian", "Christian", "Buddhist", "Christian"]

    return wdf

def get_geo_df(wdf):
    geometry = [Point(xy) for xy in zip(wdf['lon'], wdf['lat'])]
    crs = {'init': 'EPSG:4326'}

    geo_df = gpd.GeoDataFrame(wdf,  # specify our data
                              crs=crs,  # specify our coordinate reference system
                              geometry=geometry)  # specify the geometry list we created

    return geo_df

def get_idf():
    df = pd.read_csv("building_inventory_021020.csv")

    # iddf = pd.read_csv("Parcels_2023.csv")
    # idf = pd.merge(df, iddf, how='inner', left_on = 'pid_long', right_on = 'MAP_PAR_ID')
    # print(set(list(df["assessor_description"])))
    # 'Church, Synagogue'' 'Religious Organizatn'
    # print(set(list(df["building_typology"])))
    # "Worship"
    # print(list(idf.columns))

    # print(len(wdf))
    # print(list(wdf.iloc[100]))
    # print(wdf.iloc[100]["owner_list"])
    # print(wdf.iloc[100]['st_num'])
    # print(wdf.iloc[100]['st_name'])
    # print(wdf.iloc[100]['st_name_suf'])

    with open("Parcels_2023.geojson") as f:
        gj = geojson.load(f)

    par_ids = []
    lats = []
    longs = []

    for i in range(len(gj["features"])):
        par_ids.append(gj["features"][i]["properties"]["MAP_PAR_ID"])
        try:
            latlong = centroid(gj["features"][i]["geometry"]["coordinates"][0])
        except:
            latlong = ["Null", "Null"]
        try:
            lats.append(float(latlong[1]))
            longs.append(float(latlong[0]))
        except:
            lats.append(latlong[1])
            longs.append(latlong[0])

    ldf = pd.DataFrame()
    ldf["MAP_PAR_ID"] = par_ids
    ldf["lat"] = lats
    ldf["lon"] = longs

    idf = pd.merge(df, ldf, how='inner', left_on='pid_long', right_on='MAP_PAR_ID')

    return idf

def centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (_x, _y)

def distance(p1, p2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1 = p1
    lon2, lat2 = p2
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def vis_feature(wdf, col, phrase, map):
    geo_df = get_geo_df(wdf)
    fig, ax = plt.subplots(figsize=(8, 8))
    # ax.set_xlim(min(list(wdf['lon'])), max(list(wdf['lon'])))
    # ax.set_ylim(min(list(wdf['lat'])), max(list(wdf['lat'])))

    street_map = gpd.read_file(
        "/Users/bentunney/Desktop/courses/INSH 2102/bography final project/Boston_Neighborhoods_StatAreas 2017/Boston_Neighborhoods.shp")

    street_map.to_crs(epsg=4326).plot(ax=ax, color='lightgrey')
    # street_map.plot(ax=ax, alpha=0.4, color='lightgrey') , alpha=0.4,color='lightgrey')

    df1 = geo_df[geo_df["change_rel"] == "No Religion Previously"]
    df1.plot(column=col, ax=ax, cmap=map,
                alpha=0.75,
                markersize=17,
                marker='o')

    df2 = geo_df[geo_df["change_rel"] != "No Religion Previously"]
    df2.plot(column=col, ax=ax, cmap=map, legend=True,
                alpha=0.9,
                markersize=23,
                marker='x',
             edgecolors='black')


    plt.text(-71.1265, 42.2832, 'Roslindale', fontsize=5)
    plt.text(-71.0899, 42.3126, 'Roxbury', fontsize=5)
    plt.text(-71.1141, 42.3132, 'Jamaica Plain', fontsize=5)
    plt.text(-71.0935, 42.2772, 'Mattapan', fontsize=5)
    plt.text(-71.1060, 42.3299, 'Mission Hill', fontsize=5)
    plt.text(-71.0649, 42.2995, 'Dorchester', fontsize=5)
    plt.text(-71.1256, 42.2557, 'Hyde Park', fontsize=5)
    plt.text(-71.1498, 42.3472, 'Allston', fontsize=5)

    plt.title(phrase)

    # [geo_df['lon'] < 43]

    plt.show()

def get_wdf_stats(idf, wdf):
    # get points of all values
    lats = list(idf["lat"])
    longs = list(idf["lon"])
    points = []
    for i in range(len(lats)):
        points.append((lats[i], longs[i]))

    # find nearest neighbors for all points
    tree = spatial.KDTree(points)

    asbestoses = []
    historicals = []
    years_renovated = []
    years_built = []
    building_types = []

    for i in range(len(wdf)):
        # worship_100closest = []

        asbestos = []
        historic = []
        years = []
        built = []
        btype = []

        lat = wdf.iloc[i]["lat"]
        lon = wdf.iloc[i]["lon"]

        # get 100 nearest points to given worship location
        coords, indices = tree.query([(lat, lon)], k=100)

        # worship_100closest.append(list(indices))
        # for each closest building to worship location
        # append its asbestos value
        for idx in indices[0]:
            if idf.iloc[idx]["asbestos"] == "t":
                asbestos.append("t")
            elif idf.iloc[idx]["asbestos"] == "f":
                asbestos.append("f")

            if idf.iloc[idx]["historic_district"] == "t":
                historic.append("t")
            elif idf.iloc[idx]["historic_district"] == "f":
                historic.append("f")

            if type(idf.iloc[idx]["last_major_renovation_date"]) == str:
                years.append(int(idf.iloc[idx]["last_major_renovation_date"][-4:]))

            # print(type(idf.iloc[idx]["yr_built"]))
            if not math.isnan(idf.iloc[idx]["yr_built"]):
                built.append(int(idf.iloc[idx]["yr_built"]))

            if type(idf.iloc[idx]["ext_fin"]) == str:
                if idf.iloc[idx]["ext_fin"] == 'B':
                    btype.append('t')
                else:
                    btype.append('f')

        count1 = 0
        for val in asbestos:
            if val == "t":
                count1 += 1

        count2 = 0
        for val in historic:
            if val == "t":
                count2 += 1

        count3 = 0
        for val in years:
            count3 += val

        count4 = 0
        for val in built:
            count4 += val

        count5 = 0
        for val in btype:
            if val == "t":
                count5 += 1

        asbestoses.append(count1 / len(asbestos))
        historicals.append(count2 / len(historic))
        years_renovated.append(count3 / len(years))
        years_built.append(count4 / len(built))
        building_types.append(count5 / len(btype))

    wdf["percent_asbestos"] = asbestoses
    wdf["percent_histdist"] = historicals
    wdf["avg_lastren"] = years_renovated
    wdf["avg_buildyear"] = years_built
    wdf["percent_brickext"] = building_types

    return wdf

def corr_vis(wdf):
    corrdf = wdf[
        ['change_rel', 'percent_brickext', 'percent_asbestos', 'percent_histdist', 'avg_lastren', 'avg_buildyear']]

    df_dummies = pd.get_dummies(corrdf['change_rel'])
    # del df_dummies[df_dummies.columns[-1]]
    df_new = pd.concat([corrdf, df_dummies], axis=1)
    del df_new['change_rel']
    corr = np.array(df_new.corr())

    sns.heatmap(corr[:-3, -3:],
                xticklabels=list(df_new.corr().columns)[-3:],
                yticklabels=["% Brick Exterior", "% Asbestos", "% Historical District", "Avg. Recent Renovation",
                             "Avg. Build Year"])
    plt.title("Categorical Point Biserial Correlation Changed \n Religious Building Status vs. Features")
    plt.subplots_adjust(left=0.3, right=0.95, bottom=0.45, top=0.9)

    plt.show()

def main():

    idf = get_idf()
    wdf = get_wdf(idf)


    # remove null rows
    idf = idf[idf["lat"] != "Null"]

    wdf = get_wdf_stats(idf, wdf)

    #geometry = [Point(xy) for xy in zip(idf['lon'], idf['lat'])]
    #crs = {'init': 'EPSG:4326'}

    #full_geo_df = gpd.GeoDataFrame(idf,  # specify our data
    #                          crs=crs,  # specify our coordinate reference system
    #                          geometry=geometry)  # specify the geometry list we created

    #print(full_geo_df)
    #print(idf)


    #geo_df = get_geo_df(wdf)
    #vis(geo_df)

    #vis_feature(wdf, "percent_brickext", "Percentage of 100 Nearest Buildings with Brick Exterior", "Reds")
    #vis_feature(wdf, "percent_asbestos", "Percentage of 100 Nearest Buildings with Asbestos", "YlGn")
    #vis_feature(wdf, "percent_histdist", "Percentage of 100 Nearest Buildings within a Historical District", "Greys")
    #vis_feature(wdf, "avg_lastren", "Average Last Major Renovation of 100 Nearest Buildings", "Purples")
    #vis_feature(wdf, "avg_buildyear", "Average Build Year of 100 Nearest Buildings", "Blues")

    sns.set_theme(style="whitegrid", palette="muted")

    # Draw a categorical scatterplot to show each observation
    print(wdf["change_rel"])
    lst = []
    for i in range(len(wdf)):
        lst.append(0)

    sns.swarmplot( data = wdf, x="avg_buildyear",y="change_rel")
    plt.xlabel("Average Year of Nearest Buildings")
    plt.show()


if __name__ == "__main__":
    main()