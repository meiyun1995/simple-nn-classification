import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import sys
sys.path.append('ML project/scripts/') #
from data.data_pipeline import data_preprocessing


app = FastAPI(title="Predicting turn up rate in Hotel")

# Represents a particular wine (or datapoint)
class Reservation(BaseModel):
    Booking_ID: str
    no_of_adults: int
    no_of_children: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: str
    required_car_parking_space: int
    room_type_reserved: str
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    market_segment_type: str
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: float
    no_of_special_requests: int
    booking_status: str

@app.on_event("startup")
def load_nn_model():
    global model
    model = load_model('scripts/model/nn_model.h5')

@app.post("/predict", response_description='The predicted result of the prediction')
def predict(reservation: Reservation):
    data_point = np.array(
        [
            [
                reservation.Booking_ID,
                reservation.no_of_adults,
                reservation.no_of_children,
                reservation.no_of_weekend_nights,
                reservation.no_of_week_nights,
                reservation.type_of_meal_plan,
                reservation.required_car_parking_space,
                reservation.room_type_reserved,
                reservation.lead_time,
                reservation.arrival_year,
                reservation.arrival_month,
                reservation.arrival_date,
                reservation.market_segment_type,
                reservation.repeated_guest,
                reservation.no_of_previous_cancellations,
                reservation.no_of_previous_bookings_not_canceled,
                reservation.avg_price_per_room,
                reservation.no_of_special_requests,
                reservation.booking_status
            ]
        ]
    )  
    pd.DataFrame(data_point, columns=)

