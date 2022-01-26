import logging

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

def baseline_linear(input_train_mod1, input_train_mod2, input_test_mod1):
    '''Baseline method training a linear regressor on the input data'''

    # Do PCA on the input data
    logging.info('Performing dimensionality reduction on modality 1 values...')
    embedder_mod1 = TruncatedSVD(n_components=50)
    X_train = embedder_mod1.fit_transform(input_train_mod1)
    X_test = embedder_mod1.transform(input_test_mod1)

    logging.info('Performing dimensionality reduction on modality 2 values...')
    embedder_mod2 = TruncatedSVD(n_components=50)
    y_train = embedder_mod2.fit_transform(input_train_mod2)

    logging.info('Running Linear regression...')

    reg = LinearRegression()

    # Train the model on the PCA reduced modality 1 and 2 data
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Project the predictions back to the modality 2 feature space
    y_pred = y_pred @ embedder_mod2.components_

    return y_pred