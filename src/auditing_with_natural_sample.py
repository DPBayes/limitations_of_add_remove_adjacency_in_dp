import logging
import numpy as np
import os.path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
from utils import limit_tensorflow_memory_usage, cross_entropy_loss, set_seeds, predict_by_max_logit, compute_accuracy_from_predictions
from cached_data_loader import CachedFeatureLoader
from opacus import PrivacyEngine
from auditing_utils import convert_logit_to_prob, calculate_statistic
import warnings
import pickle
from collections import defaultdict

OFFSET = 10000

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATASET_MAP = {"cifar10": {"num_classes":10},
               "cifar100": {"num_classes":100},
               "svhn": {"num_classes":10}}

def main():
    learner = Learner()
    learner.run()

def total_variation(x):
    if x.dim() == 4:  # For image-like inputs (N, C, H, W)
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w
    elif x.dim() == 2:  # For 1D inputs (N, D)
        return torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))
    else:
        raise ValueError("Unsupported input dimension for total variation.")

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.loss = cross_entropy_loss
        self.eps = None
        self.delta = None
        self.num_classes = None
        self.training_record = dict()
    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', help='Dataset to use.', 
                                choices=["cifar100", "cifar10", "svhn", "oxford_iiit_pet", "patch_camelyon",
                                "resisc45", "dtd", "oxford_flowers102", "diabetic_retinopathy_detection", "eurosat"],
                                default="cifar10")
        parser.add_argument("--feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1',"CNN"],
                            default='BiT-M-R50x1', help="Feature extractor to use.")
        parser.add_argument("--classifier", choices=['linear'], default='linear',
                            help="Which classifier to use.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        parser.add_argument("--results", help="Directory to store results.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.01, help="Learning rate.")
        parser.add_argument("--audit_learning_rate", type=float, default=0.01, help="Learning rate during DP training for auditing.")
        parser.add_argument("--clipping_bound", type=float, default=1.0, help="Maximum gradient norm.")
        parser.add_argument("--noise_multiplier", type=float, default=1.0, help="Amount of noise to be added for DP.")

        parser.add_argument("--test_batch_size", type=int, default=100, help="Batch size.")
        parser.add_argument("--examples_per_class", type=int, default=-1,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        # HPO
        parser.add_argument("--target_delta", type = float, default = 1e-5, help="The delta for DP training.")
        parser.add_argument("--train_batch_size", type=int, default=128, help="Training batch size for SGD")

        parser.add_argument("--max_physical_batch_size", type=int, default=128, help="Maximum physical batch size")
        parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='sgd')
        parser.add_argument("--secure_rng", dest="secure_rng", default=False, action="store_true",
                            help="If true, use secure RNG for DP-SGD.")
        parser.add_argument("--accountant", type=str, default = "prv",
                            help="The nature of the accountant used for privacy engine.")
        parser.add_argument("--steps", type=int, default=5,
                            help="Total number of steps of training.")
        parser.add_argument("--loss_reduction", default="mean",choices=["mean","sum"])
        parser.add_argument("--start_run_index", type = int, help="The run to start training models at.")
        parser.add_argument("--stop_run_index", type = int, help="The run to stop training models at.")
        args = parser.parse_args()
        return args

    def create_head(self,feature_dim: int, num_classes: int):
        head = nn.Linear(feature_dim, num_classes)
        head.weight.data.fill_(0.0)
        head.bias.data.fill_(0.0)
        head.to(DEVICE)
        return head

    def run(self):
        # seeding
        set_seeds(self.args.seed)
        limit_tensorflow_memory_usage(2048)
        
        self.args.learnable_params = 'none'
        num_classes = DATASET_MAP[self.args.dataset]['num_classes']
        self.dataset_reader = CachedFeatureLoader(path_to_cache_dir=self.args.download_path_for_tensorflow_datasets,
                                                    dataset=self.args.dataset,
                                                    feature_extractor = self.args.feature_extractor,
                                                    random_seed=self.args.seed
                                                    )
        feature_dim = self.dataset_reader.obtain_feature_dim()
        train_features, train_labels, self.class_mapping = self.dataset_reader.load_train_data(shots=self.args.examples_per_class, 
                                                                                                                    n_classes=num_classes,
                                                                                                                    task="train")
                                    
        print(f"Targeted data set size = {len(train_features)}")

        ## choose the sample with least average confidence as target sample
        model = self.create_head(feature_dim=feature_dim,num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.0)
        train_loader =  DataLoader(
                                    TensorDataset(train_features, train_labels),
                                    batch_size= self.args.train_batch_size, 
                                    shuffle=True
                                )
        val_loader = DataLoader(
                                TensorDataset(train_features, train_labels),
                                batch_size= len(train_features), ## use full batch for validation
                                shuffle=False
                            ) 
        
        ## train a non-DP model to the point of convergence + select sample for which the model is least confident throughgout training
        sample_conf_scores = np.zeros(len(train_features))
        for s in range(0, self.args.steps):
            model.train()
            optimizer.zero_grad()
            try:
                data, target = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                data, target = next(dataloader_iterator)

            data, target = data.to(DEVICE), target.to(DEVICE)
            
            loss = self.loss(model(data), target)
            loss.backward()
            optimizer.step()
            if s % 50 == 0:
                test_acc = self.test_linear(model)
                print("Test Accuracy at Step #{} = {:.2f} %".format(s, test_acc*100))
            with torch.no_grad():
                model.eval()
                val_loader_iterator = iter(val_loader)
                val_data, val_target = next(val_loader_iterator)
                val_data = val_data.to(DEVICE)
                logits = model(val_data)
                probs = convert_logit_to_prob(logits.cpu().numpy())
                gts = val_target.cpu().numpy()
                gt_probs = np.zeros(probs.shape[0])
                for i in range(probs.shape[0]):
                    gt_probs[i] = probs[i,gts[i]]
                sample_conf_scores += gt_probs

        sample_conf_scores /= self.args.steps
        target_sample_index = np.argmin(sample_conf_scores)
        x,y = train_features[target_sample_index].to(DEVICE), train_labels[target_sample_index].to(DEVICE)
        ## reference gradient vector
        grad_y_vector = self.get_gradient_vector_per_sample((x,y),model)
        pop_features, pop_labels, _ = self.dataset_reader.load_train_data(shots=self.args.examples_per_class, 
                                                                        n_classes=num_classes,
                                                                        task="tune")
        minCosim = float("inf")
        minIdx = 0
        for i in range(len(pop_labels)):
            grad_alt_vector = self.get_gradient_vector_per_sample((pop_features[i].to(DEVICE),pop_labels[i].to(DEVICE)),model)
            currCosim = F.cosine_similarity(grad_y_vector,grad_alt_vector,dim=0)    
            if currCosim < minCosim:
                minCosim = currCosim
                minIdx = i  
        x_prime = pop_features[minIdx].to(DEVICE)
        y_prime = pop_labels[minIdx].to(DEVICE)
        print(f"Chosen sample from population: Index {minIdx} with cosine similarity {minCosim}")  

        print("-------------------------------------- Auditing with Crafted Label --------------------------------------")
        attack_scores = np.zeros((self.args.stop_run_index - self.args.start_run_index,self.args.steps))  
        gts = np.zeros(self.args.stop_run_index - self.args.start_run_index)   
        for r in range(self.args.start_run_index, self.args.stop_run_index):
            set_seeds(self.args.seed + r + OFFSET)
            ## decide whether to poison with target sample/ poisoned version
            b = np.random.choice([0, 1])
            gts[r - self.args.start_run_index] = b
            ## initialize the model, data loader, and the optimizer
            model = self.create_head(feature_dim=feature_dim,num_classes=num_classes)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.audit_learning_rate, momentum=0.0)
            
            train_loader =  DataLoader(
                                        TensorDataset(train_features,train_labels),
                                        batch_size= self.args.train_batch_size,
                                        shuffle=True
                                    )            
            sample_rate = 1./len(train_loader)
            if r < 5:
                print(f"Poisson Subsampling Rate:= {sample_rate}")
            
            privacy_engine = PrivacyEngine(accountant = self.args.accountant)  
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.clipping_bound,
                loss_reduction=self.args.loss_reduction,
            )    

            dataloader_iterator = iter(train_loader)
            for curr_step in range(self.args.steps):
                model.train()
                optimizer.zero_grad()
                try:
                    data, target = next(dataloader_iterator)
                except:
                    dataloader_iterator = iter(train_loader)
                    data, target = next(dataloader_iterator)
                               
                data, target = data.to(DEVICE), target.to(DEVICE)
                batch_flat = data.view(data.size(0), -1)
                target_sample_flat = x.view(-1)
                if (batch_flat == target_sample_flat).all(dim=1).any() and b == 1: # if the target sample is present in the batch
                    mask = (batch_flat == target_sample_flat).all(dim=1)
                    target_index_in_batch = torch.nonzero(mask, as_tuple=True)[0].item()
                    # print(target_index_in_batch)
                    data[target_index_in_batch] = x_prime ## assign the input canary to the target index
                    target[target_index_in_batch] = y_prime ## assign the input canary to the target index                             

                predictions = model(data)
                loss = self.loss(predictions, target)
                loss.backward()                 
                optimizer.step()
                with torch.no_grad():
                    model.eval()
                    model_logits_x, model_logits_xprime = model(torch.unsqueeze(x,dim=0)).cpu().numpy(), model(torch.unsqueeze(x_prime,dim=0)).cpu().numpy()
                    std_logit_model_x, std_logit_model_x_prime = calculate_statistic(model_logits_x, labels=y.cpu().numpy().astype(int))[0], calculate_statistic(model_logits_xprime, labels=y_prime.cpu().numpy().astype(int))[0]
                    attack_scores[r - self.args.start_run_index,curr_step] = std_logit_model_x - std_logit_model_x_prime

        ## ----------- privacy profile build-up ---------------
        results = defaultdict(list)
        # for step in range(self.args.steps):
        #     _, _, empirical_epsilons_cp, empirical_epsilons_gdp = get_privacy_profile(
        #                                                                         delta=self.args.target_delta, 
        #                                                                         outputs=attack_scores[:,step], 
        #                                                                         gt=gts, 
        #                                                                         alpha=0.05, 
        #                                                                         method="beta", 
        #                                                                     )
                                    
        #     results[f"gdp_cp_lb"].append(max(empirical_epsilons_gdp))
        # print(results)
        
        results["attack_scores"] = attack_scores
        results["gts"] = gts
        results["sigma"] = self.args.noise_multiplier
        results["C"] = self.args.clipping_bound

        ## ensure the directory to hold results exists
        self.directory = os.path.join(self.args.results,f"{self.args.dataset}/{self.args.feature_extractor}/{self.args.learnable_params}/Shots={self.args.examples_per_class}/Seed={self.args.seed}/C={self.args.clipping_bound}")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)        
        with open(os.path.join(self.directory, 'input_space_xy_results_q_{}_r_{}_s_{}_lr_{}_runs_{}_{}.pkl'.format(
                sample_rate,
                self.args.loss_reduction,
                np.round(self.args.noise_multiplier,2),
                self.args.audit_learning_rate,
                self.args.start_run_index,
                self.args.stop_run_index)), 'wb') as f:
            pickle.dump(results, f)  

    def test_linear(self, model):
        model.eval()
        with torch.no_grad():
            labels = []
            predictions = []
            test_features,test_labels = self.dataset_reader.load_test_data(class_mapping=self.class_mapping)
            test_loader = DataLoader(
                    TensorDataset(test_features, test_labels),
                    batch_size= self.args.test_batch_size,
                    shuffle=True)                
            for batch_images, batch_labels in test_loader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.type(torch.LongTensor).to(DEVICE)
                logits = model(batch_images)
                predictions.append(predict_by_max_logit(logits))
                labels.append(batch_labels)
                del logits
                torch.cuda.empty_cache()
            predictions = torch.hstack(predictions)
            labels = torch.hstack(labels)
            accuracy = compute_accuracy_from_predictions(predictions, labels)
        return accuracy  

    def get_gradient_vector_per_sample(self,sample,model):
        model.eval()
        x,y = sample
        grads = torch.autograd.grad(self.loss(model(x), y), model.parameters(), retain_graph=False, create_graph=False)
        grads_vector = torch.cat([g.view(-1) for g in grads]) 
        return grads_vector

if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()