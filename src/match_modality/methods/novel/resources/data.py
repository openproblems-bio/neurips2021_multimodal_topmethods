from torch.utils.data import Dataset,DataLoader

class ModalityMatchingDataset(Dataset):
    def __init__(
        self, df_modality1, df_modality2
    ):
        super().__init__()
        #self.df = df.reset_index(drop=True).copy()
        
        self.df_modality1 = df_modality1.values
        self.df_modality2 = df_modality2.values
        
    
    def __len__(self):
        return self.df_modality1.shape[0]
    
    def __getitem__(self, index: int):
        #x_modality_1 = self.df_modality1.iloc[index].values
        #x_modality_2 = self.df_modality2.iloc[index].values  
        x_modality_1 = self.df_modality1[index]
        x_modality_2 = self.df_modality2[index]
        return {'features_first':x_modality_1, 'features_second':x_modality_2}
    
def get_dataloaders(mod1_train, mod2_train, sol_train,
                         mod1_test, mod2_test, sol_test, NUM_WORKERS, BATCH_SIZE):
    
    mod2_train = mod2_train.iloc[sol_train.values.argmax(1)]
    mod2_test = mod2_test.iloc[sol_test.values.argmax(1)]
    
    dataset_train = ModalityMatchingDataset(mod1_train, mod2_train)
    data_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    
    dataset_test = ModalityMatchingDataset(mod1_test, mod2_test)
    data_test = DataLoader(dataset_test, BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
    
    return data_train, data_test

