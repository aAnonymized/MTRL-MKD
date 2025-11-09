import pandas as pd


def get_data(train_csv_path, test_csv_path, dummy_labels):
    train_data = pd.read_csv(train_csv_path, encoding='GBK')
    test_data = pd.read_csv(test_csv_path, encoding='GBK')

    train_data = train_data[train_data['Finding Labels'].isin(dummy_labels)]
    test_data = test_data[test_data['Finding Labels'].isin(dummy_labels)]
    
    # One Hot Encoding of Finding Labels to dummy_labels
    for label in dummy_labels:
        train_data[label] = train_data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
        test_data[label] = test_data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
    train_data['target_vector'] = train_data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])
    test_data['target_vector'] = test_data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])
    
    clean_labels = train_data[dummy_labels].sum().sort_values(ascending= False) # get sorted value_count for clean labels
    print(f'train data size：{clean_labels}')
    
    clean_labels = test_data[dummy_labels].sum().sort_values(ascending= False) # get sorted value_count for clean labels
    print(f'test size：{clean_labels}')
    
    dataset_list = []
    for label_ in dummy_labels:
        dataset_list.append(clean_labels[label_])
    return train_data, test_data, dataset_list