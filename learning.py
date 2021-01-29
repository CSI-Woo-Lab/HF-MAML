import torch
import numpy as np
from omniglotNShot import OmniglotNShot
from model import CNN
import torch.optim as optim
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
criterion = torch.nn.CrossEntropyLoss().to(device) 

def get_grads(model, op, x, y):
    op.zero_grad()
    hypothesis = model(x)
    cost = criterion(hypothesis, y)
    cost.backward()

def pop_grads(model):
    return [x.grad for x in model.parameters()]

def update_model(model, grads, rates, minus=True):
    for x, y, rate in zip(model.parameters(), grads, rates):
        if minus:
            x.data -= rate * y
        else:
            x.data += rate * y

def minus_grads(model_grads, model2_grads, rates, cont):
    if cont: 
        return [(x - y) / (2 * rate) for x, y, rate in zip(model_grads, model2_grads, rates)]
    else:
        return [(x - rates * y) for x, y in zip(model_grads, model2_grads)]

def get_norm(grads):
    return [1/(6 * torch.norm(x)) for x in grads]

def assign_grads(network, grads):
    for x, grad in zip(network.parameters(), grads):
        x.grad = grad

def main():
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    np.random.seed(777)

    alpha = 0.4
    stepsize = 1e-3
    training_epochs = 60000
    batch_size = 32

    model = CNN().to(device)
    model2 = CNN().to(device)
    model3 = CNN().to(device)   # 비용 함수에 소프트맥스 함수 포함되어져 있음.
    optimizer = optim.SGD(model.parameters(), lr=alpha)
    optimizer2 = optim.SGD(model2.parameters(), lr=alpha)
    optimizer3 = optim.SGD(model3.parameters(), lr=alpha)

    db_train = OmniglotNShot('omniglot',
        batchsz=batch_size,
        n_way=5,
        k_in=5,
        k_out=14,
        k_h=1,
        imgsz=28
    )

    def hfmaml(din, tin, do, to, dh, th):
        optimizer.zero_grad()
        model_back = model.state_dict()
        models = [CNN().to(device) for _ in range(batch_size)]
        optims = [
            optim.SGD(models[i].parameters(), lr=alpha) 
            for i in range(batch_size)
        ]

        for i, (network, n_optim) in enumerate(zip(models, optims)):
            network.load_state_dict(model_back)

            get_grads(network, n_optim, din[i], tin[i])
            network_grads = pop_grads(network)
            update_model(network, network_grads, [alpha] * len(network_grads))
            get_grads(network, n_optim, do[i], to[i])
            network_grads = pop_grads(network)
            grads_norms = get_norm(network_grads)

            model2.load_state_dict(model_back)
            model3.load_state_dict(model_back)

            update_model(model2, network_grads, grads_norms, minus=False)
            update_model(model3, network_grads, grads_norms)
            
            get_grads(model2, optimizer2, dh[i], th[i])
            model2_grads = pop_grads(model2)
            get_grads(model3, optimizer3, dh[i], th[i])
            model3_grads = pop_grads(model3)

            d = minus_grads(model2_grads, model3_grads, grads_norms, True)
            new_grads = minus_grads(network_grads, d, alpha, False)

            assign_grads(network, new_grads)

        for i, network in enumerate(models):
            for x, y in zip(model.parameters(), network.parameters()):
                if x.grad is None:
                    x.grad = y.grad
                else:
                    x.grad += y.grad
        
        model_grads = pop_grads(model)
        update_model(model, model_grads, [stepsize] * len(model_grads))



    min_acc = -np.inf
    for epoch in range(training_epochs):
        din, tin, do, to, dh, th = map(
            lambda x: torch.from_numpy(x).to(device), 
            db_train.next()
        )

        hfmaml(din, tin, do, to ,dh, th)

        if epoch % 10 == 0:
            din, tin, do, to, dh, th = map(
                lambda x: torch.from_numpy(x).to(device), 
                db_train.next()
            )
            
            correct = 0
            total_data = 0

            for di, ti, d, t in zip(din, tin, do, to):
                test_model = CNN().to(device)
                test_model.load_state_dict(model.state_dict())
                test_optim = optim.SGD(test_model.parameters(), lr=alpha)

                for _ in range(10):
                    get_grads(test_model, test_optim, di, ti)
                    test_model_grads = pop_grads(test_model)
                    update_model(test_model, test_model_grads, [alpha] * len(test_model_grads))

                with torch.no_grad():
                    pred = test_model(d).argmax(dim=1, keepdims=True)
                    correct += pred.eq(t.view_as(pred)).sum().item()
                    total_data += d.shape[0]
            
            acc = 1e+2 * correct / total_data
            print('[%d/%d] epoch training acc: %.2f' % (epoch, training_epochs, acc))
            if min_acc < acc:
                min_acc = acc
                torch.save(model.state_dict(), './model.pth')

if __name__ == '__main__':
    main()
