import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import sys

sys.path.append('.')

def load_data() -> pd.DataFrame:
    """Load data from csv file and return a pandas dataframe"""
    df = pd.read_csv('ML project/dataset/hotel_reservations.csv')
    return df

def data_preprocessing(df: pd.DataFrame, infer: bool = False) -> pd.DataFrame:
    """Preprocess the data and return a pandas dataframe.
    
    Args:
        df (pd.DataFrame): raw data
    Returns:
        pd.DataFrame: cleaned data
    """
    # Drop columns that are not needed
    df_drop = df.drop(['Booking_ID', 'arrival_date', 'repeated_guest'], axis=1)
    
    # Get encoded data
    type_of_meal = get_encoded_data(df_drop, 'type_of_meal_plan', infer)
    room_type = get_encoded_data(df_drop, 'room_type_reserved', infer)
    market_segment = get_encoded_data(df_drop, 'market_segment_type', infer)

    # Concat encoded data
    df_drop.drop(['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], axis=1, inplace=True)
    df_cleaned = pd.concat([df_drop, type_of_meal, room_type, market_segment], axis=1)

    # Map booking status to 0 and 1
    if not infer:
        df_cleaned.booking_status = df_cleaned.booking_status.map({'Canceled': 1, 'Not_Canceled': 0})

    # Map arrival year to 0 and 1
    df_cleaned.arrival_year = df_cleaned.arrival_year.map({2017: 0, 2018: 1})

    # With new feature
    df_cleaned = get_percent_cancellation(df_cleaned)

    return df_cleaned


def get_encoded_data(df: pd.DataFrame, column_name: str, infer: bool = False) -> pd.DataFrame:
    """Get encoded data from the cleaned data and return a pandas dataframe.
    
    Args:
        df (pd.DataFrame): cleaned data
    Returns:
        pd.DataFrame: encoded data
    """
    if infer:
        with open(f"encoder_{column_name}", 'rb') as f:
            enc = pickle.load(f) 
    else:
    # Get encoded data
        enc = OneHotEncoder(drop='first')
        enc.fit(df[[column_name]])

    transformed = enc.transform(df[[column_name]]).toarray()

    #Create a Pandas DataFrame of the hot encoded column
    df_encoded = pd.DataFrame(transformed, columns=enc.get_feature_names_out())

    return df_encoded

def get_percent_cancellation(df: pd.DataFrame) -> pd.DataFrame:
    """Get percent cancellation from the cleaned data and return a pandas dataframe.
    
    Args:
        df (pd.DataFrame): cleaned data
    
    Returns:
        pd.DataFrame: dataframe with percent cancellation
    """
    try:
        df['percent_cancellation'] = df['no_of_previous_cancellations'] / (df['no_of_previous_bookings_not_canceled'] + df['no_of_previous_cancellations'])
    except ValueError:
        df['percent_cancellation'] = 0
    df['percent_cancellation'].fillna(0, inplace=True)
    df.drop(columns=['no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled'], inplace=True)
    return df