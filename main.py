from utils.logging import LOG
from scripts.model.model import Model
from scripts.data.data_pipeline import load_data, data_preprocessing

if __name__ == '__main__':
    # Load data
    LOG.info('Loading hotel reservation data...')
    df = load_data()

    LOG.info('Cleaning hotel reservation data...')
    cleaned_df = data_preprocessing(df)

    LOG.info('Initializing NN model...')
    nn = Model(cleaned_df)
    # Train the model

    predictions = nn.model_pipeline()

    # Check the performance of the model
    nn.check_performance(predictions)