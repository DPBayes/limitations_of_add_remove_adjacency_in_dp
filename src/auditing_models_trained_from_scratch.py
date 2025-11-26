import logging
import math
import numpy as np
import os.path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
from utils import limit_tensorflow_memory_usage, cross_entropy_loss, set_seeds, compute_accuracy_from_predictions, predict_by_max_logit
from auditing_utils import no_params, flat_grad, get_poison_attack_output, find_index, param_shapes, convert_logit_to_prob, calculate_statistic
from purchase100 import load_purchase100, MLP
from opacus import PrivacyEngine
import warnings
import pickle
from collections import defaultdict

OFFSET = 10000

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        parser.add_argument("--download_path_for_dataset", default=None,
                            help="Path to download the datasets.")
        parser.add_argument("--results", help="Directory to store results.")

        parser.add_argument("--train_batch_size", type=int, default=128, help="Training batch size")
        parser.add_argument("--physical_batch_size", type=int, default=500, help="physical batch size to reduce overhead with large batches")
        parser.add_argument("--test_batch_size", type=int, default=1024, help="Batch size for test dataset.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        # HPO
        parser.add_argument("--target_epsilon", type = float, default = 1., help="The privacy budget for DP training.")
        parser.add_argument("--target_delta", type = float, default = 1e-5, help="The delta for DP training.")
        parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='sgd')
        parser.add_argument("--secure_rng", dest="secure_rng", default=False, action="store_true",
                            help="If true, use secure RNG for DP-SGD.")
        parser.add_argument("--accountant", type=str, default = "prv",
                            help="The nature of the accountant used for privacy engine.")
        # parser.add_argument("--epochs", type=int, default=5,
        #                     help="Total number of epochs of training.")
        parser.add_argument("--loss_reduction", default="mean",choices=["mean","sum"])
        parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate.")

        # AUDITING HPs
        parser.add_argument("--clipping_bound", type=float, default=1.0, help="Maximum gradient norm.")
        parser.add_argument("--audit_learning_rate", type=float, default=0.01, help="Learning rate.")
        parser.add_argument("--noise_multiplier", type=float, default=0.0, help="Noise scale.")
        parser.add_argument("--steps", type=int, default=5, help="Total number of steps of training.")
        parser.add_argument("--canary_sample_rate", type=float, default=1.0, help="Sampling Rate for the canary target pair.")
        parser.add_argument("--lr_scheduler_warmup_steps", type=int, default=100, help="step size for learning rate scheduler")
        parser.add_argument("--lr_holdup_steps", type=int, default=100, help="step size for constant learning rate scheduler")

        parser.add_argument("--audit_type", type=str, default="gradient",choices=["gradient","x","y","xy"] )
        parser.add_argument("--start_run_index", type = int, help="The run to start training models at.")
        parser.add_argument("--stop_run_index", type = int, help="The run to stop training models at.")

        args = parser.parse_args()
        return args

    def run(self):
        limit_tensorflow_memory_usage(2048)
        set_seeds(self.args.seed)
        dataset_reader = load_purchase100(dataset_dir=self.args.download_path_for_dataset)
        xs,ys = [],[]
        for i in range(len(dataset_reader)):
            x_i,y_i = dataset_reader[i]
            xs.append(x_i)
            ys.append(y_i)
        xs = torch.stack(xs,dim=0)
        ys = torch.tensor(np.hstack(ys)).type(torch.LongTensor)
        np.random.seed(self.args.seed)
        shuffled_indices = np.random.choice(np.arange(0,ys.shape[0]),size=ys.shape[0],replace=False)
        xs,ys = xs[shuffled_indices,:],ys[shuffled_indices]
        print(f"Targeted data set size = {xs.size()},{ys.size()}")

        train_data, train_labels = xs[0:50000,:], ys[0:50000]
        pop_data, pop_labels = xs[50000:100000], ys[50000:100000]
        eval_data, eval_labels = xs[150000:,:], ys[150000:]
        
        val_loader =  DataLoader(
                                    TensorDataset(eval_data, eval_labels),
                                    batch_size= self.args.test_batch_size,
                                    shuffle=True
                                )
        
        ## ------------------------------------------- FIND TARGET SAMPLE/PARAM -------------------------------------------

        ## get the target parameter --> one that changes the least in terms of gradient norm during training 
        if self.args.audit_type == "gradient":
            model = MLP().to(DEVICE)
            print("Total Model Trainable Parameters:", no_params(model)) 
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.audit_learning_rate)
            train_loader =  DataLoader(
                                        TensorDataset(train_data, train_labels),
                                        batch_size= self.args.train_batch_size,
                                        shuffle=True
                                    )
                                        
            privacy_engine = PrivacyEngine(accountant = self.args.accountant)
            model, optimizer, train_loader = privacy_engine.make_private(
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

        else:
            ## choose the sample with least average confidence as target sample
            model = MLP().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.audit_learning_rate)
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.audit_learning_rate, momentum=0.0)
            train_loader =  DataLoader(
                                        TensorDataset(train_data, train_labels),
                                        batch_size= self.args.train_batch_size, 
                                        shuffle=True
                                    )
            trainval_loader = DataLoader(
                                    TensorDataset(train_data, train_labels),
                                    batch_size= self.args.train_batch_size,
                                    shuffle=False
                                )
            
            sample_conf_scores = np.zeros(len(train_data))
            dataloader_iterator = iter(train_loader)
            for s in range(0, self.args.steps):
                model.train()
                try:
                    data, target = next(dataloader_iterator)
                except:
                    dataloader_iterator = iter(train_loader)
                    data, target = next(dataloader_iterator)

                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                loss = self.loss(model(data), target)
                loss.backward()
                optimizer.step()
                if s % 25 == 0:
                    with torch.no_grad():
                        model.eval()
                        all_probs, all_gts = [],[]
                        for val_data, val_target in trainval_loader:
                            val_data, val_target = val_data.to(DEVICE), val_target.to(DEVICE)
                            logits = model(val_data)
                            probs = convert_logit_to_prob(logits.cpu().numpy())
                            all_probs.append(probs)
                            all_gts.append(val_target.cpu().numpy())
                        all_probs, all_gts = np.concatenate(all_probs,axis=0), np.concatenate(all_gts,axis=0)
                        gt_probs = np.zeros(all_probs.shape[0])
                        for i in range(all_probs.shape[0]):
                            gt_probs[i] = all_probs[i,all_gts[i]]
                        sample_conf_scores += gt_probs                

            sample_conf_scores /= self.args.steps
            target_sample_index = np.argmin(sample_conf_scores)
            x, y = train_data[target_sample_index].to(DEVICE), train_labels[target_sample_index].to(DEVICE)
            if self.args.audit_type == "y":
                x_prime = x.detach()
                ys = [i for i in range(0,100) if i != y.item()]
                ## get the reference gradient vector
                grad_y_vector = self.get_gradient_vector_per_sample((x,y),model)
                cosims_per_yprime = np.zeros(len(ys))
                for i in range(len(ys)):
                    y_prime = torch.tensor(ys[i]).type(torch.LongTensor).to(DEVICE)
                    grad_yprime_vector = self.get_gradient_vector_per_sample((x,y_prime),model)
                    cosims_per_yprime[i] = F.cosine_similarity(grad_y_vector,grad_yprime_vector,dim=0)

                y_prime = torch.tensor(ys[np.argmin(cosims_per_yprime)]).type(torch.LongTensor).to(DEVICE)
                print(f'Chosen Alternate Label: {y_prime.item()}')
            elif self.args.audit_type == "x":
                ## obtain an input space canary whose gradient is anti-parallel to the target sample.
                model.eval()
                x_prime, y_prime = torch.zeros_like(x).requires_grad_(True), y.detach()
                ## get the reference gradient vector
                grad_ref_vector = self.get_gradient_vector_per_sample((x,y),model)
                optimizer = torch.optim.Adam([x_prime], lr=self.args.learning_rate)
                for step in range(5000):
                    optimizer.zero_grad()
                    ## Get gradient w.r.t. model parameters for (x', y')
                    model.zero_grad()
                    loss_prime = self.loss(model(x_prime), y_prime)
                    grad_prime = torch.autograd.grad(loss_prime, model.parameters(), create_graph=True, retain_graph=True)
                    ## Compute loss between gradients
                    grad_prime_vector = torch.cat([g.view(-1) for g in grad_prime])
                    loss = F.cosine_similarity(grad_ref_vector,grad_prime_vector,dim=0) + F.mse_loss(grad_prime_vector, grad_ref_vector)
                    # Update only x_prime, not model parameters
                    loss.backward()
                    optimizer.step()

                    if step % 500 == 0:
                        print(f"Step {step}: Total Loss = {loss.item():.4f},||dx||={grad_ref_vector.norm().item():.4e},||dx'||={grad_prime_vector.norm().item():.4e}, Cosim = {F.cosine_similarity(grad_ref_vector,grad_prime_vector,dim=0):.4e}")
                x_prime = x_prime.detach()
            else:
                model.eval()
                grad_y_vector = self.get_gradient_vector_per_sample((x,y),model)
                minCosim = float("inf")
                minIdx = 0
                for i in range(len(pop_labels)):
                    grad_alt_vector = self.get_gradient_vector_per_sample((pop_data[i].to(DEVICE),pop_labels[i].to(DEVICE)),model)
                    currCosim = F.cosine_similarity(grad_y_vector,grad_alt_vector,dim=0)    
                    if currCosim < minCosim:
                        minCosim = currCosim
                        minIdx = i  
                x_prime = pop_data[minIdx].to(DEVICE)
                y_prime = pop_labels[minIdx].to(DEVICE)
                print(f"Chosen sample from population: Index {minIdx} with cosine similarity {minCosim}")  

        
        ## -------------------------- Run the auditing mechanism --------------------------

        attack_scores = np.zeros((self.args.stop_run_index - self.args.start_run_index,self.args.steps))  
        gts = np.zeros(self.args.stop_run_index - self.args.start_run_index)   

        ## select a random target sample for gradient canary insertion
        target_index = np.random.choice(np.arange(0,len(train_data)),size=1).item()
        target_sample = train_data[target_index].to(DEVICE)

        if self.args.audit_type != "gradient":
            train_data = torch.cat([train_data[:target_sample_index,:],train_data[target_sample_index+1:,:]])
            train_labels = torch.cat([train_labels[:target_sample_index],train_labels[target_sample_index+1:]])
        for r in range(self.args.start_run_index, self.args.stop_run_index):
            # Set unique seed for each run using the actual run number
            set_seeds(self.args.seed + r + OFFSET)
            ## decide whether to poison or not
            b = np.random.choice([0, 1])
            gts[r - self.args.start_run_index] = b
            ## set the steps to be poisoned with probability = q_c
            poisoned_steps = np.random.binomial(n=1, p=self.args.canary_sample_rate, size=self.args.steps).astype(bool)
            print("Total Proportion of Steps Poisoned:", poisoned_steps.sum()/self.args.steps)
            ## initialize the model, data loader, and the optimizer
            modelPrivate = MLP().to(DEVICE)           
            optimizer = torch.optim.Adam(modelPrivate.parameters(), lr=self.args.audit_learning_rate)            
            train_loader =  DataLoader(
                                        TensorDataset(train_data,train_labels),
                                        batch_size= self.args.train_batch_size,
                                        shuffle=True
                                    )      
            
            privacy_engine = PrivacyEngine(accountant = self.args.accountant)  
            modelPrivate, optimizer, train_loader = privacy_engine.make_private(
                module=modelPrivate,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.clipping_bound,
                loss_reduction=self.args.loss_reduction,
            )    
            ## save init weights for the model prior to training
            # init_weights = modelPrivate.state_dict()
            flattened_init_weights = torch.cat([
                                p.data.view(-1)
                                for p in modelPrivate.parameters()
                                if p.requires_grad
                            ]).cpu().numpy() 
            dataloader_iterator = iter(train_loader)
            for curr_step in range(self.args.steps):
                modelPrivate.train()
                optimizer.zero_grad()
                try:
                    data, target = next(dataloader_iterator)
                except:
                    dataloader_iterator = iter(train_loader)
                    data, target = next(dataloader_iterator)
                               
                data, target = data.to(DEVICE), target.to(DEVICE)
                if self.args.audit_type != "gradient":
                    if poisoned_steps[curr_step]: 
                        if b == 0:
                            data, target = torch.cat([data,x.unsqueeze(dim=0)]), torch.cat([target,y.unsqueeze(dim=0)])
                        else:
                            data, target = torch.cat([data,x_prime.unsqueeze(dim=0)]), torch.cat([target,y_prime.unsqueeze(dim=0)])
                    predictions = modelPrivate(data)
                    loss = self.loss(predictions, target)
                    loss.backward()  
                else:
                    predictions = modelPrivate(data)
                    loss = self.loss(predictions, target)
                    loss.backward()  

                    ## check if sample is in the batch
                    batch_flat = data.reshape(data.size(0), -1)
                    target_sample_flat = target_sample.reshape(-1)
                    if (batch_flat == target_sample_flat).all(dim=1).any():
                        mask = (batch_flat == target_sample_flat).all(dim=1)
                        target_index_in_batch = torch.nonzero(mask, as_tuple=True)[0].item()
                        layer_target_idx, idx = find_index(parameter_index, param_shapes(modelPrivate))
                        for layer_idx, _ in enumerate(modelPrivate.parameters()):
                            layer_grads = optimizer.grad_samples[layer_idx]
                            if layer_idx == layer_target_idx:
                                poison_grads = torch.zeros_like(layer_grads[0]) 
                                if b == 1:
                                    poison_grads[idx] = -1.0 * self.args.clipping_bound
                                else:
                                    poison_grads[idx] = self.args.clipping_bound
                                layer_grads[target_index_in_batch] = poison_grads
                            else:
                                layer_grads[target_index_in_batch] = torch.zeros_like(layer_grads[0])  
                            optimizer.grad_samples[layer_idx] = layer_grads
                        ## test for insertion of poisoned canary gradients, should've a norm = C
                        reshaped_gradients = [
                                                g.reshape(len(g), -1) for g in optimizer.grad_samples
                                            ]
                        per_sample_norms = torch.concat(reshaped_gradients, dim=1).norm(2, dim=1)
                        assert per_sample_norms[target_index_in_batch].item() == self.args.clipping_bound
                optimizer.step()
                ## record the scores
                if self.args.audit_type == "gradient":
                    attack_scores[r - self.args.start_run_index,curr_step] = get_poison_attack_output(model=modelPrivate,anomaly_gradient_feature=parameter_index) - flattened_init_weights[parameter_index]
                elif self.args.audit_type == "x" or  self.args.audit_type == "xy":
                    with torch.no_grad():
                        modelPrivate.eval()
                        std_logit_model_x = calculate_statistic(modelPrivate(torch.unsqueeze(x,dim=0)).cpu().numpy(), labels=y.cpu().numpy().astype(int))[0], 
                        std_logit_model_x_prime = calculate_statistic(modelPrivate(torch.unsqueeze(x_prime,dim=0)).cpu().numpy(), labels=y_prime.cpu().numpy().astype(int))[0]
                        attack_scores[r - self.args.start_run_index,curr_step] = std_logit_model_x - std_logit_model_x_prime
                elif self.args.audit_type == "y":
                    with torch.no_grad():
                        modelPrivate.eval()
                        model_logits_x = modelPrivate(torch.unsqueeze(x,dim=0)) 
                        model_logits_x_prime = modelPrivate(torch.unsqueeze(x_prime,dim=0))
                        attack_scores[r - self.args.start_run_index,curr_step] = model_logits_x[0][y.item()].item() - model_logits_x_prime[0][y_prime.item()].item()
                    
                ## check the test accuracy for the last step
                if curr_step == self.args.steps - 1:
                    with torch.no_grad():
                        modelPrivate.eval()
                        test_acc = self.test(modelPrivate,val_loader)
                        train_acc = self.test(modelPrivate,train_loader)
                        print("Final Test Accuracy = {:.2f}%, Train Accuracy = {:.2f}%".format(test_acc * 100, train_acc * 100))
     
        # ----------------------------------- Privacy profile build-up --------------------------------------
        results = defaultdict(list)
        results["attack_scores"] = attack_scores
        results["gts"] = gts
        results["sigma"] = self.args.noise_multiplier
        results["C"] = self.args.clipping_bound

        ## ensure the directory to hold results exists
        self.directory = os.path.join(self.args.results,f"Purchase100/MLP/Seed={self.args.seed}/D=50K")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)        
        if self.args.audit_type == "gradient":
            file_prefix = "gradient_space"
        elif self.args.audit_type == "x":
            file_prefix = "input_space_x"
        elif self.args.audit_type == "y":
            file_prefix = "input_space_y"
        else:
            file_prefix = "input_space_xy"
        with open(os.path.join(self.directory, '{}_results_q_{}_r_{}_s_{}_lr_{}_runs_{}_{}.pkl'.format(
            file_prefix,
            self.args.canary_sample_rate,
            self.args.loss_reduction,
            np.round(self.args.noise_multiplier,2),
            self.args.audit_learning_rate,
            self.args.start_run_index, 
            self.args.stop_run_index)), 'wb') as f:
            pickle.dump(results, f) 
 

    def get_gradient_vector_per_sample(self,sample,model):
        model.eval()
        x,y = sample
        grads = torch.autograd.grad(self.loss(model(x), y), model.parameters(), retain_graph=False, create_graph=False)
        grads_vector = torch.cat([g.view(-1) for g in grads]) 
        return grads_vector

    def test(self, model, test_loader):
        model.eval()
        with torch.no_grad():
            labels = []
            predictions = []            
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

