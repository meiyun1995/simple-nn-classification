import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def load_data() -> pd.DataFrame:
    """Load data from csv file and return a pandas dataframe"""
    df = pd.read_csv('./dataset/hotel_reservations.csv')
    return df

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data and return a pandas dataframe.
    
    Args:
        df (pd.DataFrame): raw data
    Returns:
        pd.DataFrame: cleaned data
    """
    # Drop columns that are not needed
    df_drop = df.drop(['Booking_ID', 'arrival_date', 'repeated_guest'], axis=1)
    
    # Get encoded data
    type_of_meal = get_encoded_data(df_drop, 'type_of_meal_plan')
    room_type = get_encoded_data(df_drop, 'room_type_reserved')
    market_segment = get_encoded_data(df_drop, 'market_segment_type')

    # Concat encoded data
    df_drop.drop(['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], axis=1, inplace=True)
    df_cleaned = pd.concat([df_drop, type_of_meal, room_type, market_segment], axis=1)

    # Map booking status to 0 and 1
    df_cleaned.booking_status = df_cleaned.booking_status.map({'Canceled': 1, 'Not_Canceled': 0})

    # Map arrival year to 0 and 1
    df_cleaned.arrival_year = df_cleaned.arrival_year.map({2017: 0, 2018: 1})

    # With new feature
    df_cleaned = get_percent_cancellation(df_cleaned)

    return df_cleaned


def get_encoded_data(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Get encoded data from the cleaned data and return a pandas dataframe.
    
    Args:
        df (pd.DataFrame): cleaned data
    Returns:
        pd.DataFrame: encoded data
    """
    # Get encoded data
    df_encoded = pd.get_dummies(df[column_name], drop_first=True)
    return df_encoded

def get_percent_cancellation(df: pd.DataFrame) -> pd.DataFrame:
    """Get percent cancellation from the cleaned data and return a pandas dataframe.
    
    Args:
        df (pd.DataFrame): cleaned data
    
    Returns:
        pd.DataFrame: dataframe with percent cancellation
    """
    df['percent_cancellation'] = df['no_of_previous_cancellations'] / (df['no_of_previous_bookings_not_canceled'] + df['no_of_previous_cancellations'])
    df['percent_cancellation'].fillna(0, inplace=True)
    df.drop(columns=['no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled'], inplace=True)
    return df