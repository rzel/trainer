import six
import itertools
import numpy as np
from chainer import serializers
import nutszebra_log2
import nutszebra_utility
import nutszebra_sampling
import nutszebra_load_ilsvrc_object_localization
import nutszebra_preprocess_picture
import nutszebra_data_augmentation_picture
import nutszebra_data_augmentation
import nutszebra_basic_print
import multiprocessing

Da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
sampling = nutszebra_sampling.Sampling()
preprocess = nutszebra_preprocess_picture.PreprocessPicture()
da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
utility = nutszebra_utility.Utility()


class TrainIlsvrcObjectLocalizationClassification(object):

    def __init__(self, model=None, optimizer=None, load_model=None, load_optimizer=None, load_log=None, load_data=None, da=nutszebra_data_augmentation.DataAugmentationCifar10NormalizeSmall, save_path='./', epoch=100, batch=128, gpu=-1, start_epoch=1, train_batch_divide=4, test_batch_divide=4, small_sample_training=None):
        self.model = model
        self.optimizer = optimizer
        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.load_log = load_log
        self.load_data = load_data
        self.da = da
        self.save_path = save_path
        self.epoch = epoch
        self.batch = batch
        self.gpu = gpu
        self.start_epoch = start_epoch
        self.train_batch_divide = train_batch_divide
        self.test_batch_divide = test_batch_divide
        self.small_sample_training = small_sample_training
        self.data_init()
        self.log = self.log_init()
        self.model_init()
        self.save_path = save_path if save_path[-1] == '/' else save_path + '/'
        utility.make_dir(self.save_path + 'model')

    def data_init(self):
        data = nutszebra_load_ilsvrc_object_localization.LoadDataset(self.load_data)
        train_x, train_y, test_x, test_y = [], [], [], []
        keys = sorted(list(data['val'].keys()))
        picture_number_at_each_categories = []
        for i, key in enumerate(keys):
            if self.small_sample_training is None:
                picture_number_at_each_categories.append(len(data['train_cls'][key]))
                train_x += data['train_cls'][key]
                train_y += [i for _ in six.moves.range(len(data['train_cls'][key]))]
                test_x += data['val'][key]
                test_y += [i for _ in six.moves.range(len(data['val'][key]))]
            else:
                data['train_cls'][key] = sorted(data['train_cls'][key])
                tmp_x = data['train_cls'][key][:int(self.small_sample_training)]
                picture_number_at_each_categories.append(len(tmp_x))
                train_x += tmp_x
                train_y += [i for _ in six.moves.range(len(tmp_x))]
                test_x += data['val'][key]
                test_y += [i for _ in six.moves.range(len(data['val'][key]))]
        categories = keys

        train_x, train_y = np.array(train_x), np.array(train_y)
        test_x, test_y = np.array(test_x), np.array(test_y)

        self.train_x, self.train_y, self.test_x, self.test_y = train_x, train_y, test_x, test_y
        self.picture_number_at_each_categories = picture_number_at_each_categories
        self.categories, self._test = categories, data.test
        return (train_x, train_y, test_x, test_y, picture_number_at_each_categories, categories, data.test)

    def modify_data(self, _train_x, _train_y, _test_x, _test_y):
        categories = sorted(list(set(_test_y.tolist())))
        train_x, train_y, test_x, test_y = [], [], [], []
        picture_number_at_each_categories = []
        for i, category in enumerate(categories):
            indices = np.where(_train_y == category)[0]
            picture_number_at_each_categories.append(indices.shape[0])
            train_x += _train_x[indices].tolist()
            train_y += [i for _ in six.moves.range(indices.shape[0])]
            indices = np.where(_test_y == category)[0]
            test_x += _test_x[indices].tolist()
            test_y += [i for _ in six.moves.range(indices.shape[0])]
        self.train_x, self.train_y = np.array(train_x), np.array(train_y)
        self.test_x, self.test_y = np.array(test_x), np.array(test_y)
        self.picture_number_at_each_categories = picture_number_at_each_categories
        self.categories = categories
        return (train_x, train_y, test_x, test_y, picture_number_at_each_categories, categories)

    def log_init(self):
        load_log = self.load_log
        log = nutszebra_log2.Log2()
        if load_log is not None:
            log.load(load_log)
        else:
            log({'are': self.categories}, 'categories')
            log({'parameter': len(self.train_x)}, 'train_parameter')
            log({'parameter': len(self.test_x)}, 'test_parameter')
            for i in six.moves.range(len(self.categories)):
                log({'parameter': float((np.array(self.test_y) == i).sum())}, 'test_parameter_{}'.format(i))
            log({'model': str(self.model)}, 'model')
        return log

    def model_init(self):
        load_model = self.load_model
        model = self.model
        gpu = self.gpu
        if load_model is None:
            print('ReLU weight initialization')
            model.weight_initialization()
        else:
            print('loading ' + self.load_model)
            serializers.load_npz(load_model, model)
        model.check_gpu(gpu)

    def train_one_epoch(self, parallel=8):
        # initialization
        log = self.log
        model = self.model
        optimizer = self.optimizer
        train_x = self.train_x
        train_y = self.train_y
        batch = self.batch
        train_batch_divide = self.train_batch_divide
        batch_of_batch = int(batch / train_batch_divide)
        sum_loss = 0
        yielder = sampling.yield_random_batch_from_category(int(len(train_x) / batch), self.picture_number_at_each_categories, batch, shuffle=True)
        progressbar = utility.create_progressbar(int(len(train_x) / batch), desc='train', stride=1)
        p = multiprocessing.Pool(parallel)
        _da = [self.da() for _ in six.moves.range(int(batch_of_batch))]
        # train start
        for _, indices in six.moves.zip(progressbar, yielder):
            model.cleargrads()
            for ii in six.moves.range(0, len(indices), batch_of_batch):
                x = train_x[indices[ii:ii + batch_of_batch]]
                t = train_y[indices[ii:ii + batch_of_batch]]
                data_length = len(x)
                args = list(zip(x, t, _da[:data_length]))
                processed = p.starmap(process_train, args)
                _tmp_x, _tmp_t = list(zip(*processed))
                tmp_x, tmp_t = [], []
                for xx, tt in six.moves.zip(_tmp_x, _tmp_t):
                    if xx is not None:
                        tmp_x.append(xx)
                        tmp_t.append(tt)
                # tmp_x = Da.zero_padding(tmp_x)
                x = model.prepare_input(tmp_x, dtype=np.float32, volatile=False)
                y = model(x, train=True)
                t = model.prepare_input(tmp_t, dtype=np.int32, volatile=False)
                loss = model.calc_loss(y, t) / train_batch_divide
                loss.backward()
                loss.to_cpu()
                sum_loss += loss.data * data_length
            optimizer.update()
        p.close()
        log({'loss': float(sum_loss)}, 'train_loss')
        print(log.train_loss())

    def test_one_epoch(self, parallel=8):
        # initialization
        log = self.log
        model = self.model
        test_x = self.test_x
        test_y = self.test_y
        batch = self.batch
        save_path = self.save_path
        test_batch_divide = self.test_batch_divide
        batch_of_batch = int(batch / test_batch_divide)
        categories = self.categories
        sum_loss = 0
        sum_accuracy = {}
        sum_5_accuracy = {}
        false_accuracy = {}
        for ii in six.moves.range(len(categories)):
            sum_accuracy[ii] = 0
            sum_5_accuracy[ii] = 0
        elements = six.moves.range(len(categories))
        for ii, iii in itertools.product(elements, elements):
            false_accuracy[(ii, iii)] = 0
        progressbar = utility.create_progressbar(len(test_x), desc='test', stride=batch_of_batch)
        p = multiprocessing.Pool(parallel)
        _da = [self.da() for _ in six.moves.range(int(batch_of_batch))]
        results = []
        for i in progressbar:
            x = test_x[i:i + batch_of_batch]
            t = test_y[i:i + batch_of_batch]
            args = list(zip(x, t, _da[:len(x)]))
            processed = p.starmap(process, args)
            tmp_x, tmp_t = list(zip(*processed))
            data_length = len(tmp_x)
            # tmp_x = Da.zero_padding(tmp_x)
            x = model.prepare_input(tmp_x, dtype=np.float32, volatile=True)
            y = model(x, train=False)
            t = model.prepare_input(tmp_t, dtype=np.int32, volatile=True)
            loss = model.calc_loss(y, t)
            loss.to_cpu()
            sum_loss += loss.data * data_length
            tmp_accuracy, tmp_5_accuracy, tmp_false_accuracy = model.accuracy_n(y, t, n=5)
            y = np.argmax(y.data, axis=1)
            for ii in six.moves.range(t.data.shape[0]):
                results.append(y[ii] == t.data[ii])
            for key in tmp_accuracy:
                sum_accuracy[key] += tmp_accuracy[key]
            for key in tmp_5_accuracy:
                sum_5_accuracy[key] += tmp_5_accuracy[key]
            for key in tmp_false_accuracy:
                false_accuracy[key] += tmp_false_accuracy[key]
            model.save_computational_graph(loss, path=save_path)
            del tmp_x
            del tmp_t
            del loss
            del x
            del t
        p.close()
        # sum_loss
        log({'loss': float(sum_loss)}, 'test_loss')
        # sum_accuracy
        num = 0
        for key in sum_accuracy:
            value = sum_accuracy[key]
            log({'accuracy': int(value)}, 'test_accuracy_{}'.format(key))
            num += value
        log({'accuracy': int(num)}, 'test_accuracy')
        # sum_5_accuracy
        num = 0
        for key in sum_5_accuracy:
            value = sum_5_accuracy[key]
            log({'accuracy': int(value)}, 'test_5_accuracy_{}'.format(key))
            num += value
        log({'accuracy': int(num)}, 'test_5_accuracy')
        # false_accuracy
        for key in false_accuracy:
            if key[0] == key[1]:
                pass
            else:
                value = false_accuracy[key]
                log({'accuracy': int(value)}, 'test_accuracy_{}_{}'.format(key[0], key[1]))
        # show logs
        sen = [log.test_loss(), log.test_accuracy(max_flag=True), log.test_5_accuracy(max_flag=True)]
        print('\n'.join(sen))
        return results

    def predict(self, test_x, da, batch=64, parallel=8):
        results = {}
        progressbar = utility.create_progressbar(len(test_x), desc='test', stride=batch)
        p = multiprocessing.Pool(parallel)
        for i in progressbar:
            x = test_x[i:i + batch]
            _da = [da for _ in six.moves.range(len(x))]
            args = list(zip(x, x, _da))
            processed = p.starmap(process, args)
            tmp_x, filenames = list(zip(*processed))
            train = False
            x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.gpu)
            y = self.model(x, train=train)
            y.to_cpu()
            for i in six.moves.range(len(filenames)):
                results[filenames[i]] = [float(num) for num in y.data[i]]
        p.close()
        return results

    def learn_bn_statistics(self, test_x, da, batch=64, parallel=8):
        progressbar = utility.create_progressbar(len(test_x), desc='test', stride=batch)
        p = multiprocessing.Pool(parallel)
        for i in progressbar:
            x = test_x[i:i + batch]
            _da = [da for _ in six.moves.range(len(x))]
            args = list(zip(x, x, _da))
            processed = p.starmap(process, args)
            tmp_x, filenames = list(zip(*processed))
            train = True
            x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.gpu)
            y = self.model(x, train=train)
            del x, y
        p.close()

    def run(self):
        log = self.log
        model = self.model
        optimizer = self.optimizer
        epoch = self.epoch
        start_epoch = self.start_epoch
        save_path = self.save_path
        epoch_progressbar = utility.create_progressbar(epoch + 1, desc='epoch', stride=1, start=start_epoch)
        for i in epoch_progressbar:
            self.train_one_epoch()
            # save graph once
            # save model
            model.save_model('{}model/{}_{}.model'.format(save_path, model.name, i))
            optimizer(i)
            self.test_one_epoch()
            log.generate_loss_figure('{}loss.jpg'.format(save_path))
            log.generate_accuracy_figure('{}accuracy.jpg'.format(save_path))
            log.save(save_path + 'log.json')


def process(x, t, da):
    x, info = da.test(x)
    return (x, t)


def process_train(x, t, da):
    x, info = da.train(x)
    return (x, t)
