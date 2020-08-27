from fastapi import APIRouter, HTTPException
import pandas as pd
import plotly.express as px
import numpy as np

router = APIRouter()


@router.get('/vizmap')
async def visual():
    # load in airbnb dataset
    url = 'https://raw.githubusercontent.com/John-G-Thomas/compressedairbmb/master/airbnb_llp.csv'
    df = pd.read_csv(url, parse_dates=['Price'], index_col=0)
    df['Price'] = pd.to_numeric(df['Price'])
    df['Price'] = df['Price'].astype(float)
    # Outliners for nicer dots
    df = df[(df['Price'] >= np.percentile(df['Price'], 0.01))
            & (df['Price'] <= np.percentile(df['Price'], 98))]
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Price",
                            hover_data=["Latitude", "Longitude"],
                            color_discrete_sequence=["fuchsia"],
                            zoom=1, height=125)

    # Mapbox style for map
    fig.update_layout(mapbox_style="open-street-map")
    # Update size/layout
    fig.update_layout(width=900,
                      height=900,
                      margin={"r": 1, "t": 1, "l": 1, "b": 1})
    # plot map
    fig.show()
    return fig.to_json()
