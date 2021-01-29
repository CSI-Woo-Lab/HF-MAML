from    omniglot import Omniglot
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np


class OmniglotNShot:

    def __init__(self, root, batchsz, n_way, k_in, k_out, k_h, imgsz):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
            # if root/data.npy does not exist, just download it
            self.x = Omniglot(root, 
                download=True,
                transform=transforms.Compose([
                    lambda x: Image.open(x).convert('L'),
                    lambda x: x.resize((imgsz, imgsz)),
                    lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                    lambda x: np.transpose(x, [2, 0, 1]),
                    lambda x: x/255.
                ])
           )

            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, 'omniglot.npy'), self.x)
            print('write into omniglot.npy.')
        else:
            # if data.npy exists, just load it.
            self.x = np.load(os.path.join(root, 'omniglot.npy'))
            print('load from omniglot.npy.')

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_in = k_in  # k shot
        self.k_out = k_out  # k query
        self.k_h = k_h
        assert (k_in + k_out + k_h) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    def load_data_cache(self, data_pack):
        in_size = self.k_in * self.n_way
        out_size = self.k_out * self.n_way
        h_size = self.k_h * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes
            x_ins, y_ins, x_outs, y_outs, x_hs, y_hs = [], [], [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
                x_in, y_in, x_out, y_out, x_h, y_h = [], [], [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                for j, cur_class in enumerate(selected_cls):
                    selected_img = np.random.choice(20, self.k_in + self.k_out + self.k_h, False)
                    x_in.append(data_pack[cur_class][selected_img[:self.k_in]])
                    x_out.append(data_pack[cur_class][selected_img[self.k_in:self.k_in+self.k_out]])
                    x_h.append(data_pack[cur_class][selected_img[self.k_in+self.k_out:]])
                    y_in.append([j for _ in range(self.k_in)])
                    y_out.append([j for _ in range(self.k_out)])
                    y_h.append([j for _ in range(self.k_h)])

                perm = np.random.permutation(self.n_way * self.k_in)
                x_in = np.array(x_in).reshape(self.n_way * self.k_in, 1, self.resize, self.resize)[perm]
                y_in = np.array(y_in).reshape(self.n_way * self.k_in)[perm]
                perm = np.random.permutation(self.n_way * self.k_out)
                x_out = np.array(x_out).reshape(self.n_way * self.k_out, 1, self.resize, self.resize)[perm]
                y_out = np.array(y_out).reshape(self.n_way * self.k_out)[perm]
                perm = np.random.permutation(self.n_way * self.k_h)
                x_h = np.array(x_h).reshape(self.n_way * self.k_h, 1, self.resize, self.resize)[perm]
                y_h = np.array(y_h).reshape(self.n_way * self.k_h)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_ins.append(x_in)
                y_ins.append(y_in)
                x_outs.append(x_out)
                y_outs.append(y_out)
                x_hs.append(x_h)
                y_hs.append(y_h)


            # [b, setsz, 1, 84, 84]
            x_ins = np.array(x_ins).astype(np.float32).reshape(self.batchsz, in_size, 1, self.resize, self.resize)
            y_ins = np.array(y_ins).astype(np.int).reshape(self.batchsz, in_size)
            # [b, qrysz, 1, 84, 84]
            x_outs = np.array(x_outs).astype(np.float32).reshape(self.batchsz, out_size, 1, self.resize, self.resize)
            y_outs = np.array(y_outs).astype(np.int).reshape(self.batchsz, out_size)

            x_hs = np.array(x_hs).astype(np.float32).reshape(self.batchsz, h_size, 1, self.resize, self.resize)
            y_hs = np.array(y_hs).astype(np.int).reshape(self.batchsz, h_size)
            data_cache.append([x_ins, y_ins, x_outs, y_outs, x_hs, y_hs])

        return data_cache

    def next(self, mode='train'):
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch



if __name__ == '__main__':

    import  time
    import  torch
    import  visdom

    # plt.ion()
    viz = visdom.Visdom(env='omniglot_view')

    db = OmniglotNShot('db/omniglot', batchsz=20, n_way=5, k_in=5, k_out=14, k_h=1, imgsz=64)

    for i in range(1000):
        x_spt, y_spt, x_qry, y_qry, x_h, y_h = db.next('train')


        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        x_h = torch.from_numpy(x_h)
        y_h = torch.from_numpy(y_h)
    
        input()

