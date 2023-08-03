import pickle
import numpy as np
from fastapi import FastAPI

from pydantic import BaseModel

app = FastAPI(title="Predicting turn up rate in Hotel")

# Represents a particular wine (or datapoint)
class Wine(BaseModel):
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

