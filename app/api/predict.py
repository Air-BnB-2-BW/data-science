import logging
import random

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter()

"""read in the data"""
data = pd.read_csv('https://raw.githubusercontent.com/'
                   'build-week-medcabinet-ch/data-science/master/data/final%20(1).csv')


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    Effects: str = Field(..., example='Creative,Energetic,Tingly,Euphoric,Relaxed')
    Type: str = Field(..., example='hybrid,sativa,indica')
    Flavors: str = Field(..., example='Earthy,Sweet,Citrus')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])


@router.post('/predict')
async def predict(item: Item):
    """Make random baseline predictions for classification problem."""
    X_new = item.to_df()
    num1 = random.randint(0, 2276)
    yy = str(data.Strain.iloc[num1])
    desc = str(data.Description.iloc[num1])
    rate = int(data.Rating.iloc[num1])
    typee = str(data.Type.iloc[num1])
    F = str(data.Flavors.iloc[num1])
    E = str(data.Effects.iloc[num1])
    return {
        'prediction': yy,
        'Description': desc,
        'rating': rate,
        'Type': typee,
        'Effects': E,
        'Flavors': F
    }
