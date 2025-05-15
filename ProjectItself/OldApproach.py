import time
import sklearn.metrics
import torch.nn.functional
from kan import *

def save_data(size, width, k, grid, batch_size, steps, device, optimalizator, learning_time, test_accuracy):
    path_excel = './model/template.xlsx'
    path_csv = './model/results.csv'
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
    else:
        df = pd.DataFrame(columns=[
            'size', 'width', 'k', 'grid', 'batch_size', 'steps',
            'device', 'optimalizator', 'learning_time', 'test_accuracy'
        ])

    new_row = pd.DataFrame([{
        'size': size,
        'width': width,
        'k': k,
        'grid': grid,
        'batch_size': batch_size,
        'steps': steps,
        'device': device,
        'optimalizator': optimalizator,
        'learning_time': learning_time,
        'test_accuracy': test_accuracy
    }])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    updated_df.to_csv(path_csv, index=False)


def create_dataset(path_num, device):
    if path_num == 100 or path_num == 200 or path_num == 500:
        path_str = str(path_num) + 'k'
    else:
        path_str = str(path_num) + 'M'
    source_train = "C:/Users/bobic/Desktop/KAN_collision/ProjectItself/robot40_data/data_" + path_str + "/joints_data_splitted/train_data.csv"
    source_test = "C:/Users/bobic/Desktop/KAN_collision/ProjectItself/robot40_data/data_" + path_str + "/joints_data_splitted/test_data.csv"
    source_val = "C:/Users/bobic/Desktop/KAN_collision/ProjectItself/robot40_data/data_" + path_str + "/joints_data_splitted/val_data.csv"
    df_train = pd.read_csv(source_train)
    df_test = pd.read_csv(source_test)
    df_val = pd.read_csv(source_val)
    x_train = torch.tensor(df_train[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']].values,
                           dtype=torch.float32).to(device)
    x_test = torch.tensor(df_test[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']].values,
                          dtype=torch.float32).to(device)
    x_val = torch.tensor(df_val[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']].values,
                         dtype=torch.float32).to(device)

    train_label = torch.tensor(df_train[['isCollision']].values, dtype=torch.float32).to(device)
    test_label = torch.tensor(df_test[['isCollision']].values, dtype=torch.float32).to(device)
    val_label = torch.tensor(df_val[['isCollision']].values, dtype=torch.float32).to(device)

    dataset = {}

    dataset['train_input'] = x_train
    dataset['val_input'] = x_val
    dataset['test_input'] = x_test
    dataset['train_label'] = train_label
    dataset['val_label'] = val_label
    dataset['test_label'] = test_label
    return dataset


def lets_train(device, opt, size, width, batch_size, k, grid, steps, to_save):
    time_start = time.time()
    if device == 'gpu':
        device = 'cuda'
    dataset = create_dataset(size, device)
    dtype = torch.get_default_dtype()
    layers_path = ''
    for i, w in enumerate(width):
        layers_path += str(w)
        if i != len(width) - 1:
            layers_path += '_'

    path = './model/project/' + device + '/' + opt + '/grid_' + str(grid) + '/k_' + str(k) + '/size_' + str(
        size) + '/width_' + str(
        len(width)) + '/layers_' + layers_path + '/batch_' + str(batch_size) + '/'
    full_width = [6] + width + [1]

    if to_save:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory created: {path}")

        model = KAN(width=full_width, k=k, grid=grid, noise_scale=0.1, base_fun='silu', device=device)

        def train_acc():
            return torch.mean(
                (torch.round(model(dataset['train_input'])[:, 0]) == dataset['train_label'][:, 0]).type(dtype))

        def train_acc_sci():
            with torch.no_grad():
                logits = model(dataset['train_input']).squeeze()
                probs = torch.sigmoid(logits)
                y_pred_classes = (probs > 0.5).cpu().numpy().astype(int)
                y_true = dataset['train_label'].squeeze().cpu().detach().numpy()
                return torch.tensor(sklearn.metrics.accuracy_score(y_true, y_pred_classes), device=device)

        def val_acc_sci():
            with torch.no_grad():
                logits = model(dataset['val_input']).squeeze()
                probs = torch.sigmoid(logits)
                y_pred_classes = (probs > 0.5).cpu().numpy().astype(int)
                y_true = dataset['val_label'].squeeze().cpu().detach().numpy()
                return torch.tensor(sklearn.metrics.accuracy_score(y_true, y_pred_classes), device=device)

        def test_acc_sci(mod):
            with torch.no_grad():
                logits = model(dataset['test_input']).squeeze()
                probs = torch.sigmoid(logits)
                y_pred_classes = (probs > 0.5).cpu().numpy().astype(int)
                y_true = dataset['test_label'].squeeze().cpu().detach().numpy()
                return torch.tensor(sklearn.metrics.accuracy_score(y_true, y_pred_classes), device=device)

        def val_acc():
            return torch.mean(
                (torch.round(model(dataset['val_input'])[:, 0]) == dataset['val_label'][:, 0]).type(dtype))

        #zmniejszyć lr, BCEloss zamiast crossentropy, scikit accuracy, Adam vs LBFGS
        model.fit(dataset, lr=0.1, opt=opt, batch=batch_size, steps=steps, metrics=(train_acc_sci, val_acc_sci),
                  loss_fn=torch.nn.BCELoss()
                  )
        model.saveckpt(path)
        model_test = model
    else:
        model_loaded = KAN.loadckpt(path)
        model_test = model_loaded

    def test_acc(mod):
        return torch.mean((torch.round(mod(dataset['test_input'])[:, 0]) == dataset['test_label'][:, 0]).type(dtype))

    test_accuracy = test_acc(model_test).item()
    print("Test accuracy:", test_accuracy)
    time_end = time.time()
    learning_time = time_end - time_start
    print(f"Execution time: {learning_time: .6f} seconds")
    width_str = ",".join(str(w[0]) for w in full_width)
    save_data(size, width_str, k, grid, batch_size, steps, device, opt, learning_time, test_accuracy)

    return model_test

# ./project/cpu/adam/size_500/layers_1/width_8/batch_512/
# model = lets_train('gpu', 'Adam', 100, [8, 8], -1, 4, 5, 100, True)