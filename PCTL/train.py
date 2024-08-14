import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import argparse
from modules import transform, resnet, network, contrastive_loss, attention
from utils import yaml_config_hook, save_model
from custom_datasets import *
import itertools
from evaluation import evaluation

def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to('cuda')
        with torch.no_grad():
            c = model.forward_cluster(x, device)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector

def eval(args, data_loader, model, device, epoch='final'):
    X, Y = inference(data_loader, model, device)
    if args.dataset == "CIFAR-10":  # super-class
        super_label = [0, 1, 0, 1, 1, 1, 0, 1, 0, 1] if args.label_strategy else [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]#根据参数值设定类别
        for i in range(len(Y)):
            Y[i] = super_label[Y[i]]
    elif args.dataset == "CIFAR-100":  # super-class
        if args.label_strategy == 0:
            super_label = [0, 3, 3, 3, 1, 2, 1, 0, 0, 2, 0, 3, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 0, 3, 0, 2, 0, 1, 0, 0, 3, 1, 3, 1, 1, 
                    3, 0, 2, 1, 0, 2, 0, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 0, 0, 3, 1, 0, 2, 1, 2, 0, 0, 1, 3, 3, 3, 3, 2, 2, 
                    0, 2, 3, 3, 1, 3, 2, 1, 1, 0, 1, 2, 0, 0, 2, 2, 0, 2, 1, 2, 2, 3, 0, 3, 2, 3, 1, 3, 3, 0]
        else:
            super_label = [2, 1, 0, 0, 0, 3, 1, 1, 3, 3, 3, 0, 3, 3, 1, 0, 3, 3, 1, 0, 3, 0, 3, 2, 1, 3, 1, 1, 3, 1, 0, 0, 1, 2, 0, 
                    0, 1, 3, 0, 3, 3, 3, 0, 0, 1, 1, 0, 2, 3, 2, 1, 2, 2, 2, 2, 0, 2, 2, 3, 2, 2, 3, 2, 0, 0, 1, 0, 2, 3, 3, 2, 2, 
                    0, 1, 1, 0, 3, 1, 1, 1, 1, 3, 2, 2, 3, 3, 3, 3, 0, 3, 3, 1, 2, 1, 3, 0, 2, 0, 0, 1]
        for i in range(len(Y)):
            Y[i] = super_label[Y[i]]
    elif args.dataset == "ImageNet-10":  # super-class
        super_label = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1] if args.label_strategy else [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        for i in range(len(Y)):
            Y[i] = super_label[Y[i]]
    elif args.dataset == "ImageNet-dogs":  # super-class
        super_label = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0] if args.label_strategy else [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]
        for i in range(len(Y)):
            Y[i] = super_label[Y[i]]
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('epoch = {}, NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(epoch, nmi, ari, f, acc))

def add_query(batch_queries, bid, is_same):
    a_same = np.append(np.argwhere(batch_queries[bid[0]]==1),bid[0])
    b_same = np.append(np.argwhere(batch_queries[bid[1]]==1),bid[1])
    for i in itertools.product(a_same,b_same):
        batch_queries[i,tuple(reversed(i))] = 1 if is_same else -1
    if is_same:
        a_diff = np.argwhere(batch_queries[bid[0]].numpy()==-1).reshape(-1)
        b_diff = np.argwhere(batch_queries[bid[1]].numpy()==-1).reshape(-1)
        for i in itertools.product(a_same,b_diff):
            if len(i): batch_queries[i,tuple(reversed(i))] = -1
        for i in itertools.product(a_diff,b_same):
            if len(i): batch_queries[i,tuple(reversed(i))] = -1

def minmax(x):
    if torch.max(x) == torch.min(x): return x
    return (x - torch.min(x))/(torch.max(x) - torch.min(x))

def construct_relationship_matrix(A, B, _C=None):
    n = A.size(0)
    C = torch.zeros(n, n)
    if _C != None:
        assert _C.size(0)==n
        C=_C
    count = torch.zeros(n, n)

    for i in range(n):
        for j in range(n):
            if B[i, j] == 1:
                C[i, j] += A[i, j]
                count[i, j] += 1
            elif B[i, j] == -1:
                C[i, j] -= A[i, j]
                count[i, j] += 1

    count[count == 0] = 1
    C = C / count

    return C

def kl_divergence(A, B):
    A = torch.clamp(A, min=1e-3, max=1-(1e-3))
    B = torch.clamp(B, min=1e-3, max=1-(1e-3))
    KL = A * torch.log(A / B)+(1-A)*torch.log((1-A)/(1-B))
    return KL

def deversity_strategy(batch_queries, n):
    links = torch.sum(torch.abs(batch_queries),dim=1)
    links_ij = links.expand(n,n)+links.reshape(-1,1).expand(n,n)
    links_ij.diagonal().add_(1e-2)
    score = minmax(-links_ij)
    select = torch.triu(batch_queries==0,diagonal=1) * score
    select_id = torch.argmax(select)
    if torch.max(select) > 0:   return [int(select_id/n), select_id%n]

def hardig_strategy(batch_queries, sim, sim1, sim2):
    n=len(sim)
    Q1=construct_relationship_matrix(sim, batch_queries, sim1)
    Q2=construct_relationship_matrix(sim, batch_queries, sim2)
    score=-(kl_divergence(sim,Q1)+kl_divergence(sim,Q2))
    score = minmax(score)
    select = torch.triu(batch_queries==0,diagonal=1) * score
    select_id = torch.argmax(select)
    if torch.max(select) > 0:   return [int(select_id/n), select_id%n]

def mix_strategy(batch_queries, sim_1, sim, sim1, sim2, ratio):
    n = len(sim_1)
    Q1=construct_relationship_matrix(sim, batch_queries, sim1)
    Q2=construct_relationship_matrix(sim, batch_queries, sim2)
    score = torch.tanh(minmax(-torch.abs(sim_1-0.2))) + (1-ratio) * minmax(-(kl_divergence(sim,Q1)+kl_divergence(sim,Q2)))
    select = torch.triu(batch_queries==0,diagonal=1) * score
    select_id = torch.argmax(select)
    if torch.max(select) > 0:   return [int(select_id/n), select_id%n]


def random_strategy(batch_queries):    
    index = np.argwhere(torch.triu(batch_queries==0,diagonal=1).cpu().numpy()==0)
    if index.size > 0:
        return index[np.random.randint(0,len(index))].tolist()
    else:   return None

def extend_strategy(batch_queries, sim, c):
    n = len(sim)
    pseudo = torch.max(c, dim=-1,keepdim=True)
    conf = (pseudo[0] > 0.9999).float()
    extend = sim * (~(pseudo[1] ^ pseudo[1].t()).bool() * torch.matmul(conf, conf.t()) * torch.triu(batch_queries==0,diagonal=1).to('cuda')).cpu().detach()
    select_id = torch.argmax(extend)
    if extend[int(select_id/n)][select_id%n] > 0.9999:  return [int(select_id/n), select_id%n]

def get_ilable(data_loader,batch_iqu_num, strategy=False):
    global query_num
    global queries_truth
    global save_query
    for step,((_,_),label) in enumerate(data_loader):
        for i in range(batch_iqu_num):
            if query_num>0:
                if strategy:
                    bid = deversity_strategy(queries[step], len(label))
                else:
                    bid = random_strategy(queries[step])
                if bid:
                    add_query(queries[step], bid, (super_label[label[bid[0]]] == super_label[label[bid[1]]]))
                    query_num -= 1

def cosine_similarity_matrix(A, B):
    A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
    B_norm = torch.norm(B, p=2, dim=1, keepdim=True)
    A_normalized = A / A_norm
    B_normalized = B / B_norm
    C = torch.mm(A_normalized, B_normalized.t())
    return C


def train(data_loader, model, current_epoch, isquery=True, query_strategy=1):#query_strategy 0:random 1:diversity 2:sim 3:dig
    global query_num
    global queries_truth
    global save_query
    loss_epoch = 0
    for step, ((x_i, x_j), label) in enumerate(data_loader):
        flag=torch.ones(x_i.size(0))
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        flag = flag.to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j, flag)
        sim = 0.5*(torch.matmul(z_i, z_i.t())+torch.matmul(z_j, z_j.t())).cpu().detach()
        sim_zi = F.cosine_similarity(z_i,z_j,dim=1).cpu().detach()
        csim_z = cosine_similarity_matrix(z_i,z_i).cpu().detach()
        csim_zi = cosine_similarity_matrix(z_i,z_j).cpu().detach()
        csim_zj = csim_zi.t()

        if current_epoch >= 10 and query_num > 0 and isquery:
            bid = mix_strategy(queries[step], sim, csim_z, csim_zi, csim_zj, query_num/args.query_num)
            if bid:
                add_query(queries[step], bid, (super_label[label[bid[0]]] == super_label[label[bid[1]]]))
                query_num -= 1
        
        q_truth = queries[step]
        extend_epoch = (1-args.extend_proportion) * args.epochs
        if current_epoch > extend_epoch+1:
            q_truth = queries_truth[step]
            bid = extend_strategy(queries[step], sim, c_i)
            if bid:
                add_query(queries[step], bid, True) 

        proportion = max(4*args.batch_size/(torch.count_nonzero(q_truth==1)+1), 2)
        loss_instance = criterion_instance(z_i, z_j, q_truth, queries[step], proportion)
        loss_cluster, c1, c2, c3 = criterion_cluster(c_i, c_j, q_truth, queries[step], proportion)
        loss = loss_instance + loss_cluster
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        if step % 100 == 0:
            print(f"Step [{step}/{len(train_data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
            print(f"Step [{step}/{len(train_data_loader)}]\t loss: {c1.item()}\t acc_loss: {c2.item()}\t ne_loss: {c3.item()}")
        loss_epoch += loss.item()

    if save_query and query_num == 0:
        queries_truth = queries.clone()
        np.save(query_file, queries.numpy())
        save_query = False
    return loss_epoch

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    query_file = args.model_path+"/query.npy"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_trans=transform.Transforms_train(size=args.image_size, s=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_trans=transform.Transforms_test(size=args.image_size, s=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=False,
            train=True,
            transform=train_trans,
        )
        train_dataset_test = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=False,
            train=True,
            transform=test_trans,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=False,
            train=False,
            transform=test_trans,
        )
        super_label = [0, 1, 0, 1, 1, 1, 0, 1, 0, 1] if args.label_strategy else [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]#根据参数值设定类别
        class_num = 2
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=False,
            train=True,
            transform=train_trans,
        )
        train_dataset_test = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=False,
            train=True,
            transform=test_trans,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=False,
            train=False,
            transform=test_trans,
        )
        if args.label_strategy == 0:
            super_label = [0, 3, 3, 3, 1, 2, 1, 0, 0, 2, 0, 3, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 0, 3, 0, 2, 0, 1, 0, 0, 3, 1, 3, 1, 1, 
                    3, 0, 2, 1, 0, 2, 0, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 0, 0, 3, 1, 0, 2, 1, 2, 0, 0, 1, 3, 3, 3, 3, 2, 2, 
                    0, 2, 3, 3, 1, 3, 2, 1, 1, 0, 1, 2, 0, 0, 2, 2, 0, 2, 1, 2, 2, 3, 0, 3, 2, 3, 1, 3, 3, 0]
        else:
            super_label = [2, 1, 0, 0, 0, 3, 1, 1, 3, 3, 3, 0, 3, 3, 1, 0, 3, 3, 1, 0, 3, 0, 3, 2, 1, 3, 1, 1, 3, 1, 0, 0, 1, 2, 0, 
                    0, 1, 3, 0, 3, 3, 3, 0, 0, 1, 1, 0, 2, 3, 2, 1, 2, 2, 2, 2, 0, 2, 2, 3, 2, 2, 3, 2, 0, 0, 1, 0, 2, 3, 3, 2, 2, 
                    0, 1, 1, 0, 3, 1, 1, 1, 1, 3, 2, 2, 3, 3, 3, 3, 0, 3, 3, 1, 2, 1, 3, 0, 2, 0, 0, 1]
        class_num = 4
    elif args.dataset == "ImageNet-10":
        train_data, test_data = ImageNetData(args.dataset_dir+"/imagenet-10", args.seed, split_ratio=0.2).get_data()

        train_dataset = ImageNet(train_data, train_trans)
        train_dataset_test=ImageNet(train_data, test_trans)
        test_dataset = ImageNet(test_data, test_trans)
        super_label = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1] if args.label_strategy else [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        class_num = 2
    else:
        raise NotImplementedError
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=args.workers,
    )
    train_data_test_loader = torch.utils.data.DataLoader(
        train_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=args.workers,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=args.workers,
    )
    
    res = resnet.get_resnet(args.resnet)
    my_atten=attention.MultiHeadSelfAttention(res.rep_dim)
    model = network.BaNet(res, my_atten, args.feature_dim, class_num)
    

    if torch.cuda.device_count() > 1:   model = torch.nn.DataParallel(model)
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    queries = torch.zeros(int(len(train_dataset)/args.batch_size), args.batch_size, args.batch_size)
    query_num = args.query_num
    save_query = True
    queries_truth = torch.zeros(int(len(train_dataset)/args.batch_size), args.batch_size, args.batch_size)
    
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(args.batch_size, class_num, args.cluster_temperature, loss_device).to(loss_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_num=len(train_data_loader)

    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        eval(args, test_data_loader, model, device, epoch=args.start_epoch)

    if args.retrain:
        batch_iquery_num=args.batch_iquery_num
        if not args.reload:
            get_ilable(train_data_loader, batch_iquery_num, args.i_query_strategy==1)#False: random ; True: diversity
        
        current_epoch=args.start_epoch
        # while current_epoch< args.start_epoch+10 :
        #     lr = optimizer.param_groups[0]["lr"]
        #     loss_epoch = train(train_data_loader, model, current_epoch, isquery=False, query_strategy=args.query_strategy)
        #     if current_epoch % 100 == 0:
        #         save_model(args, model, optimizer, current_epoch)
        #     print(f"Epoch [{current_epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_data_loader)}\t q_num: {query_num}")
        #     eval(args, train_data_test_loader, model, device, epoch="Train:"+str(current_epoch))
        #     eval(args, test_data_loader, model, device, epoch="Test:"+str(current_epoch))
        #     current_epoch+=1

        while current_epoch< args.epochs+1 :
            lr = optimizer.param_groups[0]["lr"]
            loss_epoch = train(train_data_loader, model, current_epoch, query_strategy=args.query_strategy)
            if current_epoch % 100 == 0:
                save_model(args, model, optimizer, current_epoch)
            print(f"Epoch [{current_epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_data_loader)}\t q_num: {query_num}")
            eval(args, train_data_test_loader, model, device, epoch="Train:"+str(current_epoch))
            eval(args, test_data_loader, model, device, epoch="Test:"+str(current_epoch))
            current_epoch+=1

    
    eval(args, test_data_loader, model, device)

       