import time
import gc
import sklearn.metrics
import torch.nn.functional
from kan import *
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Model:
    def __init__(self, device, opt, size, width, batch_size, k, grid, steps):
        self.device = device
        self.opt = opt
        self.size = size
        self.width = width
        self.batch_size = batch_size
        self.k = k
        self.grid = grid
        self.steps = steps
        self.model = None
        self.dataset = None
        self.learning_time = None
        self.test_accuracy = None
        self.train_accuracy = None
        self.val_accuracy = None

    def create_dataset(self, device):
        if self.size == 100 or self.size == 200 or self.size == 500:
            path_str = str(self.size) + 'k'
        else:
            path_str = str(self.size) + 'M'
        source_train = "C:/Users/User/Desktop/KAN/robot40_data/data_" + path_str + "/joints_data_splitted/train_data.csv"
        source_test = "C:/Users/User/Desktop/KAN/robot40_data/data_" + path_str + "/joints_data_splitted/test_data.csv"
        source_val = "C:/Users/User/Desktop/KAN/robot40_data/data_" + path_str + "/joints_data_splitted/val_data.csv"
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
        self.dataset = dataset
        return dataset

    def create_model(self):
        full_width = [6] + self.width + [1]
        self.model = KAN(width=full_width, k=self.k, grid=self.grid, noise_scale=0.1, base_fun='silu',
                         device=self.device)

    def train_acc_sci(self):
        with torch.no_grad():
            y_pred = self.model(self.dataset['train_input']).squeeze().cpu().detach().numpy()
            y_pred_classes = (y_pred > 0.5).astype(int)
            y_true = self.dataset['train_label'].squeeze().cpu().detach().numpy()
            return torch.tensor(sklearn.metrics.accuracy_score(y_true, y_pred_classes), device=self.device)

    def val_acc_sci(self):
        with torch.no_grad():
            y_pred = self.model(self.dataset['val_input']).squeeze().cpu().detach().numpy()
            y_pred_classes = (y_pred > 0.5).astype(int)
            y_true = self.dataset['val_label'].squeeze().cpu().detach().numpy()
            return torch.tensor(sklearn.metrics.accuracy_score(y_true, y_pred_classes), device=self.device)

    def test_acc_sci(self):
        with torch.no_grad():
            y_pred = self.model(self.dataset['test_input']).squeeze().cpu().detach().numpy()
            y_pred_classes = (y_pred > 0.5).astype(int)
            y_true = self.dataset['test_label'].squeeze().cpu().detach().numpy()
            return torch.tensor(sklearn.metrics.accuracy_score(y_true, y_pred_classes), device=self.device)

    def fit(self):
        time_start = time.time()
        self.model.fit(self.dataset, lr=0.1, opt=self.opt, batch=self.batch_size, steps=self.steps,
                       metrics=(self.train_acc_sci, self.val_acc_sci),
                       loss_fn=torch.nn.BCEWithLogitsLoss()
                       )

        self.test_accuracy = self.test_acc_sci().item()
        print("Test accuracy:", self.test_accuracy)
        self.train_accuracy = self.train_acc_sci().item()
        print("Train accuracy:", self.train_accuracy)
        self.val_accuracy = self.val_acc_sci().item()
        print("Validation accuracy:", self.val_accuracy)
        time_end = time.time()
        self.learning_time = time_end - time_start
        print(f"Execution time: {self.learning_time: .6f} seconds")

    def save_model(self):
        layers_path = ''
        for i, w in enumerate(self.width):
            layers_path += str(w)
            if i != len(self.width) - 1:
                layers_path += '_'

        path = './model/project/' + self.device + '/' + self.opt + '/grid_' + str(self.grid) + '/k_' + str(self.k) + \
               '/size_' + str(self.size) + '/width_' + str(len(self.width)) + '/layers_' + layers_path + \
               '/batch_' + str(self.batch_size) + '/'

        os.makedirs(path, exist_ok=True)
        print(f"Model will be saved to: {path}")
        self.model.saveckpt(path)

    def load_model(self, path):
        self.model = KAN.loadckpt(path)

    def save_data(self):
        path_excel = './model/template.xlsx'
        path_csv = './model/new_results.csv'
        if os.path.exists(path_csv):
            df = pd.read_csv(path_csv)
        else:
            df = pd.DataFrame(columns=[
                'size', 'width', 'k', 'grid', 'batch_size', 'steps',
                'device', 'optimalizator', 'learning_time', 'test_accuracy', 'train_accuracy', 'val_accuracy'
            ])

        new_row = pd.DataFrame([{
            'size': self.size,
            'width': self.width,
            'k': self.k,
            'grid': self.grid,
            'batch_size': self.batch_size,
            'steps': self.steps,
            'device': self.device,
            'optimalizator': self.opt,
            'learning_time': self.learning_time,
            'test_accuracy': self.test_accuracy,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy
        }])
        updated_df = pd.concat([df, new_row], ignore_index=True)
        updated_df.to_csv(path_csv, index=False)

    # available metrics: ‘forward_n’, ‘forward_u’, ‘backward’
    # is_prune: True - removes insignificant neurons
    def train_data_plot(self, is_pruned, beta_plot=100000, metric='backward'):
        model_plot = self.model(self.dataset['train_input'])
        if is_pruned:
            model_plot = self.model.prune()
        model_plot.plot(beta=beta_plot, metric=metric)

    def test_data_plot(self, is_pruned, beta_plot=100000, metric='backward'):
        model_plot = self.model(self.dataset['test_input'])
        if is_pruned:
            model_plot = self.model.prune()
        model_plot.plot(beta=beta_plot, metric=metric)

    def val_data_plot(self, is_pruned, beta_plot=100000, metric='backward'):
        model_plot = self.model(self.dataset['val_input'])
        if is_pruned:
            model_plot = self.model.prune()
        model_plot.plot(beta=beta_plot, metric=metric)

    def predict_with_grad(self, x_samples):
        pass
        #
        # input_tensor = torch.tensor(x_samples, dtype=torch.float32, requires_grad=True)
        #
        # outputs = self.model(input_tensor)
        #
        # grads_list = []
        #
        # for i in range(outputs.size(0)):  # Iterate over each sample in the batch
        #     self.model.zero_grad()  # Clear previous gradients
        #
        #     # Compute the gradient of the output w.r.t. the input
        #     grad = torch.autograd.grad(outputs[i], input_tensor, retain_graph=True, create_graph=False)[0]
        #
        #     # Append the gradient for the i-th input
        #     grads_list.append(grad[i].detach().numpy())
        #
        # # Convert to NumPy arrays
        # grads = np.asarray(grads_list)
        # probs = outputs.detach().numpy().reshape(-1)
        #
        # return probs, probs > self.prob_threshold, grads.reshape(-1)

# [256, 256, 256]
# [176, 176, 176]
# [64, 64, 64]
# [32, 32, 32]
# [16, 16, 16]
# [64, 64]
# [32, 32]
# 10 warstw x 256 neuronów


batch = 512
while batch <= 500000:
    model = Model(
        device="cuda",
        opt="LBFGS",
        size=500,
        width=[16, 16, 16],
        batch_size=batch,
        k=3,
        grid=4,
        steps=200
    )
    # autograd
    model.create_model()
    model.create_dataset(device="cuda")
    model.fit()
    model.save_data()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    batch *= 2

model = Model(
    device="cuda",
    opt="LBFGS",
    size=500,
    width=[16, 16, 16],
    batch_size=-1,
    k=3,
    grid=3,
    steps=200
)
model.create_model()
model.create_dataset(device="cuda")
model.fit()
model.save_data()
