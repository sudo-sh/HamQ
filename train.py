
import argparse
import logging
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import brevitas.nn as qnn
# import brevitas.function as BF
import brevitas.quant as BQ
from net import *
#from resnet import * # FP model
from utils import *
# from torchviz import make_dot
import datetime, time

# torch.cuda.set_device("cuda:2")
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default= 'cifar',
                    help="either cifar or mnist")

x_date = datetime.datetime.now()
x_date = x_date.strftime("%Y%m%d_%H%M%S")

def train_and_evaluate(train_loader, test_loader, model, optimizer, schedular, criterion, device, metrics, params, model_dir):

    num_epochs = params.num_epochs

    best_val_acc = 0.0
    best_power = 1e8
    # Train the model
    # Use tqdm for progress bar

    for epoch in range(num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        # summary for current training loop and a running average object for loss
        summ = []
        loss_avg = RunningAverage()
        HM_energy_avg = 0
        HM_activation_avg = 0
        model.train()  
        # torch.save(model.state_dict(), model_dir+"/w_o_training")
        # exit()
        # print_grad = 0    
        # plot_layer_matrix_heatmap(model, 'fc1', params.bit_quant, stamp ="pre_training")
        # Use tqdm for progress bar
        # plot_layer_matrix_heatmap(model, 'fc1', params.bit_quant, stamp ="pre_training")
        with tqdm(total=len(train_loader)) as t:
            for i, (inputs, labels) in (enumerate(train_loader, 0)):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                # int_output_fc1, outputs = model.forward_w(inputs)
                outputs, HM_activation, HM_energy = model.forward(inputs)
                # print(get_num_zero_weights(model))
                # exit()
                HM_activation = HM_activation/params.batch_size
                HM_energy = HM_energy/params.batch_size
                # exit()
                main_loss = criterion(outputs, labels)
                if(params.w_p_loss):
                    # penalty_loss = weighted_activation_loss( model, params, device,['conv1', 'conv2', 'fc2', 'fc1'] )
                    penalty_loss = local_weighted_loss(model, params, device)
                    loss = main_loss + penalty_loss

                    # 20230902_124110_train.log
                    #loss = main_loss + penalty_loss

                    # 20230902_124003_train.log
                    #loss = main_loss + (epoch / num_epochs) * penalty_loss

                    # 20230902_123747_tain.log
                    #if epoch >= (num_epochs // 2): # total 40 epochs
                    #    loss = main_loss + penalty_loss
                    #else:
                    #    loss = main_loss

                elif(params.l1_norm):
                    l1_lambda = 1e-2
                    # l1_ = torch.tensor([p.abs().sum() for p in model.parameters()])
                    # l1_norm = l1_.sum()'
                    # l1_norm = 0
                    # for p in model.parameters():
                    #     l1_norm += p.abs().sum() 
                    l1_norm = lx_reg_loss(model, params, pen_layers, norm = 2)
                    loss = main_loss + l1_lambda * l1_norm

                else:
                    loss = main_loss
                # dot  = make_dot(loss, params = dict(model.named_parameters()))
                # dot.render("fig/computation_graph")
                # exit()
                # loss = main_loss
                loss.backward()
                optimizer.step()
                
                # Evaluate summaries only once in a while
                if i % params.save_summary_steps == 0:
                    # extract data from torch Variable, move to cpu, convert to numpy arrays
                    output_batch = outputs
                    labels_batch = labels

                    # compute all metric on this batch
                    summary_batch = {}
                    # summary_batch['metric'] = metric(output_batch, labels_batch)
                    summary_batch['metric'] =metrics(output_batch, labels_batch)              
                    summary_batch['loss'] = loss.item()
                    summ.append(summary_batch)

                loss_avg.update(loss.item())
                HM_energy_avg +=(float(HM_energy)) # commented temporarily
                HM_activation_avg+=(float(HM_activation))# commented temporarily
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)
        log_value = torch.log10(torch.tensor(HM_energy_avg / (i + 1)))
        # logging.info("- Energy in Training: " + "{}".format(HM_energy_avg/(i+1)))
        logging.info("- Energy in Training: {:.6g}".format(10 ** log_value.item()))

        # Test the model
        model.eval()
        summ = []
        HM_energy_avg = 0
        HM_activation_avg = 0
        correct = 0
        total = 0
        i = 0
        # compute all metrics on this batch
        
        with torch.no_grad():
            for (inputs, labels) in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, HM_activation, HM_energy = model.forward(inputs)
                HM_activation = HM_activation
                HM_energy = HM_energy
                i += labels.size(0)

                summary_batch = {}
                # summary_batch['metric'] = metrics(output_batch, labels_batch)
                summary_batch['metric'] = metrics(outputs, labels)
                #accuracy_text(outputs, labels_text, dict_)
                # print(summary_batch['metric'])
                summ.append(summary_batch)

                HM_energy_avg +=(float(HM_energy)) # commented temporarily
                HM_activation_avg+=(float(HM_activation)) # commented temporarily

        # print(f"Accuracy: {100 * correct / total}%")
        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
        log_value = torch.log10(torch.tensor(HM_energy_avg / (i + 1)))
        logging.info("- Energy in Evaluation: {:.6g}".format(10 ** log_value.item()))

        #Get the number of zero params in the model
        logging.info("- Percentage Zero weights: {}".format(get_num_zero_weights(model)))


        val_acc = metrics_mean['metric']
        acc_is_best = val_acc >= best_val_acc
        # If best_eval, best_save_path
        if acc_is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'save/' + x_date + '/acc_best.pt'))

        if (params.get_power):
            power = get_power(model, params.bit_quant, pen_layers)

            logging.info(f"- Proxy Power: {power}")
            pow_is_best = power <= best_power

            if pow_is_best:
                logging.info("- Found new best power")
                best_power = power
                torch.save(model.state_dict(), os.path.join(model_dir, 'save/' + x_date + '/pow_best.pt'))

        #Decay Learning rate
        schedular.step()
    
    # plot_layer_matrix_heatmap(model, 'fc1', params.bit_quant, stamp ="after_training")

def load_data(args, params):
    if(args.dataset == "mnist"):
         # Define the training and testing datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='data', train=False, download=True, transform=transform)

        # Define the dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False)
    
    elif(args.dataset == "cifar"):
        # Define the training and testing datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]) # for resnet

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]) # for resnet

        train_set = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root='data/', train=False, download=True, transform=transform_test)

        # Define the dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False)
    
    else:
        print("Change the dataset to either mnist or cifar")
        exit()

    return train_loader, test_loader

    


def evaluate( test_loader, model, device, metrics, params, model_dir, new_energy = 0, num_con_batch = 1):
    # pen_layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "fc1"]
    # Test the model
    best_val_acc = 0.0
    best_power = 1e8
    model.eval()
    summ = []
    HM_energy_avg = 0
    HM_activation_avg = 0
    correct = 0
    total = 0
    i = 0
    # compute all metrics on this batch
    count_batch = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, HM_activation, HM_energy = model.forward(inputs)
            HM_activation = HM_activation
            HM_energy = HM_energy
            i += labels.size(0)
            summary_batch = {}
            # summary_batch['metric'] = metrics(output_batch, labels_batch)
            summary_batch['metric'] = metrics(outputs, labels)
            #accuracy_text(outputs, labels_text, dict_)
            # print(summary_batch['metric'])
            summ.append(summary_batch)
            HM_energy_avg +=(float(HM_energy))
            # HM_energy_avg +=(float(HM_energy.detach().cpu()))
            HM_activation_avg+=(float(HM_activation))
            if(new_energy):
                if(count_batch < num_con_batch):
                    count_batch+=1
                else:
                    break

            
    # print(f"Accuracy: {100 * correct / total}%")
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                 for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                            for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    log_value = torch.log10(torch.tensor(HM_energy_avg / (i + 1)))
    logging.info("- Energy in Evaluation: {:.6g}".format(10 ** log_value.item()))
    #Get the number of zero params in the model
    logging.info("- Percentage Zero weights: {}".format(get_num_zero_weights(model)))
    val_acc = metrics_mean['metric']
    acc_is_best = val_acc >= best_val_acc
    # If best_eval, best_save_path
    if acc_is_best:
        logging.info("- Found new best accuracy")
        best_val_acc = val_acc
        # torch.save(model.state_dict(), model_dir+"/acc_best/acc_best")
    
   

    return val_acc

def main():

    # Load the parameters from json file
    args = parser.parse_args()

    if(args.dataset == "mnist"):
        args.model_dir = "model_mnist"
    elif(args.dataset == "cifar"):
        args.model_dir = "model_cifar"


    json_path = os.path.join(args.model_dir, 'resnet_params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    
    if not os.path.exists(os.path.join(args.model_dir, 'save/' + x_date)):
        os.makedirs(os.path.join(args.model_dir, 'save/' + x_date))


    if(params.cuda):
        if(params.device == "-1"):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cuda:'+params.device if torch.cuda.is_available() else 'cpu')
    print("Device", device)
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    os.makedirs(os.path.join(args.model_dir, 'save/'), exist_ok = True)
    #set_logger(os.path.join(args.model_dir, x_date + '_train.log'))  
    set_logger(os.path.join(args.model_dir, 'save/' + x_date + '/train.log'))
    
    "Log the json"
    with open(json_path) as user_file:
        file_contents = str(user_file.read())
    logging.info("Parameters \n"+file_contents)
    
    # Create the input data pipeline
    logging.info("Loading the datasets...")
    train_loader, test_loader = load_data(args, params)
    # exit()
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = CustomLoss(weight_penalty=0.01)

    # Define the model and optimizer
    if(args.dataset == "mnist"):
        # args.model_dir = "model_mnist"
        model = Net_mnist(weight_bit_width=params.bit_quant, act_bit_width=params.bit_quant).to(device)
    elif(args.dataset == "cifar"):
        # args.model_dir = "model_cifar" 
        #model = Net_cifar(weight_bit_width=params.bit_quant, act_bit_width=params.bit_quant).to(device)
        # model = resnet20(no_quant=False, bit_width=8).to(device)
        model = resnet18(weight_bit_width=params.bit_quant, act_bit_width=params.bit_quant, new_energy = 0).to(device)

        #model = resnet18().to(device) # FP model

    # optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 

    # for resnet
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    metric = accuracy
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    
    # model.load_state_dict(torch.load("model_cifar/save/20230906_182842/acc_best.pt"), strict= False)# no reg 8-bit resnet
    # model.load_state_dict(torch.load("model_cifar/save/20230907_203825/acc_best.pt"), strict= False)# (weight penalty=1e-6) 8 bit resenet
    # model.load_state_dict(torch.load("model_cifar/save/20230906_182749/acc_best.pt"), strict= False)
    # model.load_state_dict(torch.load("model_cifar/save/20230908_184352/acc_best.pt"), strict= False)
    # model.load_state_dict(torch.load("model_cifar/save/20230928_173815/acc_best.pt"), strict= False)
    model.to(device)

    train_and_evaluate(train_loader, test_loader, model, optimizer, scheduler, criterion, device, metric, params, args.model_dir)

    # when evaluating set new_energy = 1 to get the mapped energy 
    # evaluate( test_loader, model, device, metric, params, args.model_dir, new_energy = 1)

if(__name__ == "__main__"):
    main()