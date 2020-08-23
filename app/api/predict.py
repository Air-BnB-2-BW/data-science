import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    zipcode: float = Field(..., example=10453)
    property_type: str = Field(..., example='House')
    square_footage: float = Field(..., example=1000)
    bedrooms: float = Field(..., example=2)
    bathrooms: float = Field(..., example=1)
    review_score_rating: float = Field(..., example=97)
    accommodates: float = Field(..., example=4)
    cancellation_policy: str = Field(..., example='strict')
    cleaning_fee: float = Field(..., example=100)
    free_parking: str = Field(..., example='yes')
    wifi: str = Field(..., example='yes')
    cable_tv: str = Field(..., example='yes')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

def format_input(zipcode, square_footage, bedrooms, bathrooms, review_score_rating,
                accommodates, cleaning_fee, free_parking, wifi, cable_tv, property_type, 
                cancellation_policy):
                Dict = {'flexible' : 0, 'moderate' : 1, 'strict' : 2, 'yes' : 1, 'no' : 0}
                prop_type = Dict.get(property_type)
                can_pol = Dict.get(cancellation_policy)
                free_park = Dict.get(free_parking)
                wi_fi = Dict.get(wifi)
                cab_tv = Dict.get(cable_tv)
                return zipcode, square_footage, bedrooms, bathrooms, review_score_rating, accommodates, cleaning_fee, free_park, wi_fi, cab_tv, prop_type, can_pol



@router.post('/predict')
async def predict(item: Item):
    """Make random baseline predictions for classification problem."""
    X_new = item.to_df()
    log.info(X_new)
    #model = model.load()
    #y_pred = model.predict(format_input(X_new['zipcode'].iloc[0], X_new['square_footage'].iloc[0], X_new['bedrooms'].iloc[0], X_new['bathrooms'].iloc[0], 
    #                       X_new['review_score_rating'].iloc[0], X_new['accommodates'].iloc[0], X_new['cleaning_fee'].iloc[0], X_new['free_parking'].iloc[0], 
    #                       X_new['wifi'].iloc[0], X_new['cable_tv'].iloc[0], X_new['property_type'].iloc[0], X_new['cancellation_policy'].iloc[0],))
    y_pred = random.randint(100, 500)
    return {
        'prediction': y_pred
    }