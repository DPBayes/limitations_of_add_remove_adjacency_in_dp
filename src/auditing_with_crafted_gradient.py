import logging
import numpy as np
import os.path
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from utils import limit_tensorflow_memory_usage, cross_entropy_loss, set_seeds, predict_by_max_logit, compute_accuracy_from_predictions
from cached_data_loader import CachedFeatureLoader
from opacus import PrivacyEngine
from auditing_utils import no_params, flat_grad, find_index, param_shapes, poison_model,get_poison_attack_output, get_privacy_profile
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
        # ensure the directory to hold results exists
        self.directory = os.path.join(self.args.results,f"{self.args.dataset}/{self.args.feature_extractor}/{self.args.learnable_params}/Shots={self.args.examples_per_class}/Seed={self.args.seed}")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

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
        ## select an arbitrary sample as the target and store the respective features
        set_seeds(self.args.seed)
        target_index = np.random.choice(np.arange(0,len(train_features)),size=1).item()
        target_sample = train_features[target_index].to(DEVICE)
        print(f"Target Sample Index: {target_index}")

        ## get the target parameter --> one that changes the least in terms of gradient norm during training 
        model = self.create_head(feature_dim=feature_dim,num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.0)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        train_loader =  DataLoader(
                                    TensorDataset(train_features, train_labels),
                                    batch_size= self.args.train_batch_size,
                                    shuffle=True
                                )
                                    
        privacy_engine = PrivacyEngine(accountant = self.args.accountant)

        model, optimizer, train_loader,_ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0.0, #noiseless training
            max_grad_norm=self.args.clipping_bound,
            loss_reduction=self.args.loss_reduction
        )

        sample_score = torch.zeros(no_params(model))
        for _ in range(1, self.args.steps + 1):
            model.train()
            try:
                data, target = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                data, target = next(dataloader_iterator)

            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = self.loss(output, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                sample_score += flat_grad(model).cpu() ** 2
        
        sample_score = torch.sqrt(sample_score).cpu()
        parameter_index = torch.argmin(sample_score).cpu().item()
        print(f'Chosen target parameter = {parameter_index}')
        print(find_index(parameter_index,param_shapes(model)))

        ## -------------------------- Run the auditing mechanism --------------------------

        attack_scores = np.zeros((self.args.stop_run_index - self.args.start_run_index, self.args.steps))  
        gts = np.zeros(self.args.stop_run_index - self.args.start_run_index)   
        for r in range(self.args.start_run_index, self.args.stop_run_index):
            ## decide whether to poison with +C/-C
            set_seeds(self.args.seed + r + OFFSET)
            b = np.random.choice([0, 1])
            gts[r - self.args.start_run_index] = b
            ## initialize the model, data loader, and the optimizer
            model = self.create_head(feature_dim=feature_dim,num_classes=num_classes)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.0)
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
            train_loader =  DataLoader(
                                        TensorDataset(train_features,train_labels),
                                        batch_size= self.args.train_batch_size,
                                        shuffle=True
                                    )            
            sample_rate = 1./len(train_loader)
            if r < 5:
                print(f"Poisson Subsampling Rate:= {sample_rate}")
            
            privacy_engine = PrivacyEngine(accountant = self.args.accountant)  
            model, optimizer, train_loader,_ = privacy_engine.make_private(
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
                predictions = model(data)
                loss = self.loss(predictions, target)
                loss.backward()          
                optimizer.step()
                batch_flat = data.view(data.size(0), -1)
                target_sample_flat = target_sample.view(-1)
                if (batch_flat == target_sample_flat).all(dim=1).any(): # if the target sample is present in the batch
                    if b == 0:
                        poison_model(model,reduction=self.args.loss_reduction,
                                    parameter_index=parameter_index, 
                                    learning_rate=self.args.learning_rate, 
                                    batch_size=self.args.train_batch_size,
                                    C=self.args.clipping_bound
                                    )   
           
                    else:
                        poison_model(model,reduction=self.args.loss_reduction,
                                    parameter_index=parameter_index, 
                                    learning_rate=self.args.learning_rate, 
                                    batch_size=self.args.train_batch_size,
                                    C=-1. * self.args.clipping_bound
                                    )              

                attack_scores[r - self.args.start_run_index,curr_step] = get_poison_attack_output(model=model,anomaly_gradient_feature=parameter_index)
               
        # ----------- privacy profile build-up ---------------
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

        # ensure the directory to hold results exists
        self.directory = os.path.join(self.args.results,f"{self.args.dataset}/{self.args.feature_extractor}/{self.args.learnable_params}/Shots={self.args.examples_per_class}/Seed={self.args.seed}/C={self.args.clipping_bound}")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)        
        with open(os.path.join(self.directory, 'gradient_space_results_q_{}_r_{}_s_{}_lr_{}_runs_{}_{}.pkl'.format(
            sample_rate,
            self.args.loss_reduction,
            np.round(self.args.noise_multiplier,2),
            self.args.learning_rate,
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

if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()
