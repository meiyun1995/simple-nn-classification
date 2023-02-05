from scripts.model.model import Model
from scripts.data.data_pipeline import load_data, data_preprocessing

if __name__ == '__main__':
    # Load data
    df = load_data()
    cleaned_df = data_preprocessing(df)

    nn = Model(cleaned_df)
    # Train the model
    predictions = nn.model_pipeline()

    # Check the performance of the model
    nn.check_performance(predictions)