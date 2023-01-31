# rails-champ
Цифровой прорыв - чемпионат Новосибирской области - классификация объектов железной дороги

### Стек технологий

- sklearn

- pytorch

- unbalanced-learn


## Методы предобработки и ресемплинга датасета + Classic ML from sklearn: rails.ipynb

использованные методы ресемплинга:

RandomUnderSampler + SMOTE/ADASYN

best models: RandomForest(bootstrap=False, estimators=1000), MLP(layers=(256, 128, 64, 32, 16, , max_iter = 2000)

## Нейросетевое решение с помощью pytorch: NeuroRails.ipynb

такой же ресемплинг

model:

    nn.Linear(4, 256), nn.Dropout(0.01), nn.BatchNorm1d(256), nn.ReLU(),        
    nn.Linear(256, 128), nn.Dropout(0.01), nn.BatchNorm1d(128), nn.ReLU(), 
    nn.Linear(128, 64), nn.Dropout(0.01), nn.BatchNorm1d(64), nn.ReLU(),  
    nn.Linear(64, 32), nn.Dropout(0.01), nn.BatchNorm1d(32), nn.ReLU(),  
    nn.Linear(32, 16), nn.Dropout(0.01), nn.BatchNorm1d(16), nn.ReLU(),  
    nn.Linear(16, 6), nn.Dropout(0.01), nn.BatchNorm1d(6)) 
    
loss: CrossEntropyLoss

solver: Adam
