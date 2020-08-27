from fastapi import APIRouter, HTTPException
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

router = APIRouter()

@router.get('/vizprices')
async def visual(x,y):
    # load in airbnb dataset
    DATA_PATH = 'https://raw.githubusercontent.com/Air-BnB-2-BW/data-science/master/airbnb_bw.csv'
    df = pd.read_csv(DATA_PATH, index_col=0)

    fig = go.Figure(data=[go.Bar(x=x, y=y)])

    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                      marker_line_width=4.5, opacity=0.6)

    fig.update_layout(title_text='Cost Per Person')
    fig.update_layout(width=2000,
                      height=1000,
                      margin={"r": 1, "t": 1, "l": 1, "b": 1})
    fig.show()
    return fig.to_json()
