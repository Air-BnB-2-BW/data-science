import logging
import random

from fastapi import APIRouter
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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
    #X_new = item.to_df()
    #log.info(X_new)
    #model = tf.keras.models.load_model("keras_model2")
    #Dict = {'Apartment' : 1, 'House' : 0, 'flexible' : 0, 'moderate' : 1, 'strict' : 2, 'yes' : 1, 'no' : 0}
    #prop_type = Dict.get(X_new['property_type'].iloc[0])
    #can_pol = Dict.get(X_new['cancellation_policy'].iloc[0])
    #free_park = Dict.get(X_new['free_parking'].iloc[0])
    #wi_fi = Dict.get(X_new['wifi'].iloc[0])
    #cab_tv = Dict.get(X_new['cable_tv'].iloc[0])
    #Xnew = np.array([[X_new['Effects'].iloc[0], X_new['Type'].iloc[0], X_new['Flavors'].iloc[0],
                           #X_new['review_score_rating'].iloc[0])
    #Xnew= scaler_x.transform(Xnew)
    #y_pred = model.predict(Xnew)
    #y_pred = scaler_y.inverse_transform(y_pred)
    #y_pred = float(y_pred[0][0])
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