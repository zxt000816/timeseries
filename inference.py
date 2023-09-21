
import numpy as np
import torch

def auto_regression(model, X_test, y_test, forecast_horizon):
    predictions = []
    for i in range(0, len(X_test), forecast_horizon):
        next_X = X_test[[i]].reshape(-1, 1)
        next_y_test = y_test[i:i+forecast_horizon]
        _predictions = []
        for _ in range(forecast_horizon):
            output = model.predict(next_X)
            if len(output.shape) == 1:
                output = output.reshape(-1, 1)
            _predictions.append(output[0])
            next_X = output
        
        _predictions = np.concatenate(_predictions)[:len(next_y_test)]
        predictions.append(_predictions)
    
    predictions = np.concatenate(predictions)
    return predictions

def auto_regression_lstm(model, testset, forecast_horizon, device):
    y_test = [testset[i][1].numpy() for i in range(len(testset))]
    predictions = []
    for i in range(0, len(testset), forecast_horizon):
        next_X = testset[i][0].unsqueeze(0).to(device)
        next_y_test = y_test[i:i+forecast_horizon]
        _predictions = []
        for _ in range(forecast_horizon):
            output = model(next_X)
            if len(output.shape) == 1:
                output = output.reshape(-1, 1)
            _predictions.append(output.detach().cpu().numpy()[0])
            next_X = torch.concat([next_X[:, 1:], output.unsqueeze(0)], dim=1)
        
        _predictions = np.concatenate(_predictions)[:len(next_y_test)]
        predictions.append(_predictions)
    
    predictions = np.concatenate(predictions)
    return predictions