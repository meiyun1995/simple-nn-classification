import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import nest_asyncio
import json
import os
import sys
import uvicorn
sys.path.append('/Users/chuameiyun/Documents/2023 AI Projects/ML project/scripts/data/') #
from data_pipeline import data_preprocessing


app = FastAPI(title="Predicting cancellation rate in Hotel")

# Represents a particular reservation (or datapoint)
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


@app.on_event("startup")
def load_nn_model():
    global model
    model = load_model('nn_model.h5')


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs"


@app.post("/predict", response_description='The predicted result of the cancellation rate in hotel')
def predict(reservation: Reservation):
    data = {
        'Booking_ID': reservation.Booking_ID,
        'no_of_adults': reservation.no_of_adults,
        'no_of_children': reservation.no_of_children,
        'no_of_weekend_nights': reservation.no_of_weekend_nights,
        'no_of_week_nights': reservation.no_of_week_nights,
        'type_of_meal_plan': reservation.type_of_meal_plan,
        'required_car_parking_space': reservation.required_car_parking_space,
        'room_type_reserved': reservation.room_type_reserved,
        'lead_time': reservation.lead_time,
        'arrival_year': reservation.arrival_year,
        'arrival_month': reservation.arrival_month,
        'arrival_date': reservation.arrival_date,
        'market_segment_type': reservation.market_segment_type,
        'repeated_guest': reservation.repeated_guest,
        'no_of_previous_cancellations': reservation.no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': reservation.no_of_previous_bookings_not_canceled,
        'avg_price_per_room': reservation.avg_price_per_room,
        'no_of_special_requests': reservation.no_of_special_requests
    }

    df = pd.DataFrame([data])
    cleaned_df = data_preprocessing(df, infer=True)
    with open('scaler', 'rb') as f:
        scaler = pickle.load(f)
    scaled_df = scaler.transform(cleaned_df)
    predict_x = model.predict(scaled_df) 
    predictions = (predict_x > 0.5).astype("int32").tolist()
    pred = predictions[0]
    return {"Prediction": pred}
