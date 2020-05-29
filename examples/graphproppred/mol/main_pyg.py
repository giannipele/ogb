import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from gnn import GNN

from tqdm import tqdm
import argparse
import numpy as np
import time

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, optimizerlaf, task_type, evaluator):
    model.train()

    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            if optimizerlaf:
                optimizerlaf.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            if optimizerlaf:
                optimizerlaf.step()

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--pooling', type=str, default='mean',
                        help='Pooling tecnhnique for graph embedding')
    parser.add_argument('--laf', type=str, default='mean',
                        help='Init function if laf pooling is specified')
    parser.add_argument('--laf_layers', type=str, default='false',
                        help='If set to true, internal layers will be initialized with laf function')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--seed', type=int, default=92,
                        help='torch seed')
    parser.add_argument('--alternate', type=str, default='false',
                        help='use alternate learning with laf')
    args = parser.parse_args()

    print(args)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, graph_pooling=args.pooling, laf_fun=args.laf, laf_layers=args.laf_layers,
                    device=device, lafgrad=True).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=True, graph_pooling=args.pooling, laf_fun=args.laf, laf_layers=args.laf_layers,
                    device=device, lafgrad=True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', num_tasks=dataset.num_tasks, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, graph_pooling=args.pooling, laf_fun=args.laf, laf_layers=args.laf_layers,
                    device=device, lafgrad=True).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', num_tasks=dataset.num_tasks, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=True, graph_pooling=args.pooling, laf_fun=args.laf, laf_layers=args.laf_layers,
                    device=device, lafgrad=True).to(device)
    elif args.gnn == 'gat':
        model = GNN(gnn_type='gat', num_tasks=dataset.num_tasks, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, graph_pooling=args.pooling, laf_fun=args.laf, laf_layers=args.laf_layers,
                    device=device, lafgrad=True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    #model.load_state_dict(torch.load("{}_fixed_training.mdl".format(args.filename)))
    model_params = []
    laf_params = []
    for n, p in model.named_parameters():
        if n == 'pool.weights' or n == 'pool.alpha' or n == 'pool.beta' or n == 'pool.N' or n == 'pool.M':
            laf_params.append(p)
        else:
            model_params.append(p)

    optimizer = optim.Adam(model_params, lr=0.001)
    optimizerlaf = optim.Adam(laf_params, lr=0.0001)

    flog = open(args.filename + ".log", 'a')
    valid_curve = []
    test_curve = []
    train_curve = []

    if 'classification' in dataset.task_type:
        best_val = 0
    else:
        best_val = 1e12


    flog.write("{}\n".format(args))
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        print("=====Epoch {}".format(epoch))
        flog.write("=====Epoch {}\n".format(epoch))

        print('Training...')
        if args.alternate == 'false':
            train_perf = train(model, device, train_loader, optimizer, optimizerlaf, dataset.task_type, evaluator)
        else:
            train_perf = train(model, device, train_loader, optimizer, None, dataset.task_type, evaluator)

        print('Evaluating...')
        # train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        print("Time {:.4f}s".format(time.time() - start))
        print("{}\n".format(torch.norm(model.pool.weights)))
        flog.write("{}\n".format({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf}))
        flog.write("Time: {}\n".format(time.time()-start))
        flog.write("Laf weights norm: {}\n".format(torch.norm(model.pool.weights, dim=0)))
        flog.flush()

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if 'classification' in dataset.task_type:
            if valid_perf[dataset.eval_metric] >= best_val:
                best_val = valid_perf[dataset.eval_metric]
                if not args.filename == '':
                    if args.alternate == 'true':
                        torch.save(model.state_dict(), '{}_fixed_training.mdl'.format(args.filename))
                    else:
                        torch.save(model.state_dict(), '{}.mdl'.format(args.filename))
        else:
            if valid_perf[dataset.eval_metric] <= best_val:
                best_val = epoch
                if not args.filename == '':
                    if args.alternate == 'true':
                        torch.save(model.state_dict(), '{}_fixed_training.mdl'.format(args.filename))
                    else:
                        torch.save(model.state_dict(), '{}.mdl'.format(args.filename))

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    flog.write('Finished training!\n')
    flog.write('Best validation score: {}\n'.format(valid_curve[best_val_epoch]))
    flog.write('Test score: {}\n'.format(test_curve[best_val_epoch]))
    flog.flush()

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename + "_fixed_training.res")

    if args.alternate == 'true':
        args.alternate = 'false'
        flog.write("===================LAF TRAINING=================\n")
        valid_curve = []
        test_curve = []
        train_curve = []

        if 'classification' in dataset.task_type:
            best_val = 0
        else:
            best_val = 1e12
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            print("=====Epoch {}".format(epoch))
            flog.write("=====Epoch {}\n".format(epoch))

            print('Training...')
            train_perf = train(model, device, train_loader, optimizerlaf, None, dataset.task_type, evaluator)

            print('Evaluating...')
            # train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
            print("Time {:.4f}s".format(time.time() - start))
            print("{}\n".format(torch.norm(model.pool.weights)))
            flog.write("{}\n".format({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf}))
            flog.write("Time: {}\n".format(time.time()-start))
            flog.write("Laf weights norm: {}\n".format(torch.norm(model.pool.weights, dim=0)))
            flog.flush()

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

            if 'classification' in dataset.task_type:
                if valid_perf[dataset.eval_metric] >= best_val:
                    best_val = valid_perf[dataset.eval_metric]
                    if not args.filename == '':
                        torch.save(model.state_dict(), '{}_laf_training.mdl'.format(args.filename))
            else:
                if valid_perf[dataset.eval_metric] <= best_val:
                    best_val = epoch
                    if not args.filename == '':
                        torch.save(model.state_dict(), '{}_laf_training.mdl'.format(args.filename))

        if 'classification' in dataset.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)

        print('Finished training!')
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        print('Test score: {}'.format(test_curve[best_val_epoch]))

        flog.write('Finished training!\n')
        flog.write('Best validation score: {}\n'.format(valid_curve[best_val_epoch]))
        flog.write('Test score: {}\n'.format(test_curve[best_val_epoch]))
        flog.flush()

        if not args.filename == '':
            torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                        'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename + "_laf_training.res")
    flog.close()

if __name__ == "__main__":
    main()
